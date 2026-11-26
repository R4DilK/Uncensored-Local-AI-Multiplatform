import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'wakelock_service.dart';
import 'chat_storage_service.dart';
import 'log_service.dart';

/// Wraps llamadart's LlamaEngine for model loading, generation, and lifecycle.
class LlmService extends GetxService {
  LlamaEngine? _engine;
  LlamaBackend? _backend;

  final isLoaded = false.obs;
  final isGenerating = false.obs;
  final loadedModelPath = ''.obs;
  final tokensPerSecond = 0.0.obs;
  final lastGenerationTokens = 0.obs;
  final lastGenerationSpeed = 0.0.obs;

  // ── Loading progress tracking ──────────────────────────────
  final isLoadingModel = false.obs;
  final loadingProgress = 0.0.obs; // 0.0 to 1.0
  final loadingStatusMsg = ''.obs;
  bool _loadingCancelled = false;

  StreamSubscription? _generateSub;

  // ── Ollama (cloud/local server) mode ───────────────────────
  /// When true, generation is sent to an Ollama server instead of the
  /// local llama.cpp engine (e.g. `deepseek-v4.1-flash:cloud`).
  final ollamaActive = false.obs;
  final ollamaModel = ''.obs;
  String _ollamaBaseUrl = 'http://127.0.0.1:11434';
  http.Client? _ollamaClient;
  bool _ollamaStopRequested = false;

  String get ollamaBaseUrl => _ollamaBaseUrl;

  String get loadedModelFilename {
    final path = loadedModelPath.value;
    if (path.isEmpty) return '';
    return p.basename(path);
  }

  String get publicModelId {
    final filename = loadedModelFilename;
    if (filename.isEmpty) return 'local';
    final stem = filename.toLowerCase().endsWith('.gguf')
        ? filename.substring(0, filename.length - 5)
        : p.basenameWithoutExtension(filename);
    return stem
        .replaceAll(RegExp(r'[^A-Za-z0-9._-]+'), '-')
        .replaceAll(RegExp(r'-+'), '-')
        .replaceAll(RegExp(r'^-|-$'), '');
  }

  /// Initialize the service.
  Future<LlmService> init() async {
    // Backend is created fresh per loadModel() call — no init needed here
    return this;
  }

  /// Cancel an in-progress model load.
  void cancelLoading() {
    _loadingCancelled = true;
  }

  /// Load a GGUF model from [path] with progress tracking.
  Future<void> loadModel(String path) async {
    LogService? log;
    try { log = Get.find<LogService>(); } catch (_) {}

    // Verify file exists first
    final file = File(path);
    if (!await file.exists()) {
      log?.error('Model file not found: $path', source: 'LLM');
      throw Exception('Model file not found: $path');
    }

    final filename = p.basename(path);
    log?.info('Loading model: $filename', source: 'LLM');

    _loadingCancelled = false;
    isLoadingModel.value = true;
    loadingProgress.value = 0.0;
    loadingStatusMsg.value = 'Preparing...';

    // Enable wake lock during model loading (heavy memory operation)
    WakelockService? wakelockService;
    try {
      wakelockService = Get.find<WakelockService>();
    } catch (_) {}

    // Unload previous if any — MUST fully tear down engine + backend
    if (_engine != null || isLoaded.value) {
      loadingStatusMsg.value = 'Unloading previous model...';
      loadingProgress.value = 0.05;
      await _fullTeardown();
      // Give native side time to release resources
      await Future.delayed(const Duration(milliseconds: 500));
      if (_loadingCancelled) {
        _resetLoadingState();
        return;
      }
    }

    // Fresh backend + engine for every load — prevents stale native state
    // Wrapped in try-catch to handle SELinux crashes on Android where
    // ggml_backend_load_all() attempts to scan '/' which is denied.
    try {
      _backend = LlamaBackend();
      _engine = LlamaEngine(_backend!);
    } catch (e) {
      _backend = null;
      _engine = null;
      _resetLoadingState();
      log?.error('Engine init failed: $e', source: 'LLM');
      throw Exception(
        'Failed to initialize AI engine. '
        'This may be a device compatibility issue. '
        'Error: $e',
      );
    }

    try {
      loadingStatusMsg.value = 'Loading into memory...';
      loadingProgress.value = 0.1;

      // Get file size for display
      final fileSize = await file.length();
      final sizeGb = (fileSize / (1024 * 1024 * 1024)).toStringAsFixed(1);
      loadingStatusMsg.value = 'Loading $sizeGb GB into memory...';

      // Start a timer to animate progress while loading
      Timer? progressTimer;
      progressTimer = Timer.periodic(const Duration(milliseconds: 300), (
        timer,
      ) {
        if (_loadingCancelled) {
          timer.cancel();
          return;
        }
        // Gradually increase progress (asymptotic approach to 0.95)
        final current = loadingProgress.value;
        if (current < 0.95) {
          loadingProgress.value = current + (0.95 - current) * 0.04;
        }
      });

      if (_loadingCancelled) {
        progressTimer.cancel();
        await _fullTeardown();
        _resetLoadingState();
        return;
      }

      // Use smaller context on Android to prevent OOM kills.
      // Desktop can handle 2048, but Android devices with limited RAM
      // need 1024 to avoid the Low Memory Killer (LMK).
      // Desktop GPUs have plenty of room: 8K tokens of conversation memory.
      final contextSize = Platform.isAndroid ? 1024 : 8192;

      // Map the string backend to GpuBackend enum
      final storage = Get.find<ChatStorageService>();
      GpuBackend parsedBackend;
      switch (storage.backendType) {
        case 'vulkan':
          parsedBackend = GpuBackend.vulkan;
          break;
        case 'opencl':
          parsedBackend = GpuBackend.opencl;
          break;
        default:
          parsedBackend = GpuBackend.cpu;
      }

      // Read gpu layers
      final userGpuLayers = storage.gpuLayers;

      // Threads: mobile keeps 4 for memory stability; desktop uses roughly
      // one thread per physical core (logical cores / 2), between 4 and 16.
      final cores = Platform.numberOfProcessors;
      final int threads;
      if (Platform.isAndroid || Platform.isIOS) {
        threads = cores > 4 ? 4 : 0;
      } else {
        threads = (cores ~/ 2).clamp(4, 16);
      }
      final params = ModelParams(
        contextSize: contextSize,
        gpuLayers: userGpuLayers,
        preferredBackend: parsedBackend,
        numberOfThreads: threads,
        numberOfThreadsBatch: threads,
      );

      log?.info('Backend=$parsedBackend, GPU layers=$userGpuLayers, ctx=$contextSize, threads=$threads', source: 'LLM');

      await _engine!.loadModel(path, modelParams: params);
      progressTimer.cancel();

      if (_loadingCancelled) {
        // User cancelled while loading — full cleanup
        await _fullTeardown();
        _resetLoadingState();
        return;
      }

      loadingProgress.value = 1.0;
      loadingStatusMsg.value = 'Ready!';
      isLoaded.value = true;
      loadedModelPath.value = path;
      log?.info('Model loaded successfully: $filename', source: 'LLM');

      // Enable wake lock for inference on mobile (keeps app from being killed)
      final modelName = p.basenameWithoutExtension(path);
      await wakelockService?.enableForInference(modelName: modelName);

      // Brief delay to show 100%
      await Future.delayed(const Duration(milliseconds: 300));
    } catch (e) {
      isLoaded.value = false;
      loadedModelPath.value = '';
      await _fullTeardown();
      log?.error('Model load failed: $e', source: 'LLM');

      // Provide a clearer error message for common Android failures
      if (Platform.isAndroid) {
        final errStr = e.toString().toLowerCase();
        if (errStr.contains('memory') || errStr.contains('alloc')) {
          throw Exception(
            'Not enough RAM to load this model. '
            'Try a smaller model (e.g. Gemma 2 2B at 1.6 GB).',
          );
        }
      }
      rethrow;
    } finally {
      _resetLoadingState();
    }
  }

  void _resetLoadingState() {
    isLoadingModel.value = false;
    loadingProgress.value = 0.0;
    loadingStatusMsg.value = '';
    _loadingCancelled = false;
  }

  /// Tokens/patterns the model may emit that should be stripped from output.
  /// Covers ChatML, Llama, Gemma, Phi, Mistral, and other common formats.
  static final _stopPatterns = RegExp(
    r'<\|end\|>'
    r'|<\|eot_id\|>'
    r'|<\|endoftext\|>'
    r'|<\|im_end\|>'
    r'|<\|im_start\|>'
    r'|<end_of_turn>'
    r'|<start_of_turn>'
    r'|<\|assistant\|>'
    r'|<\|user\|>'
    r'|<\|system\|>'
    r'|<\|pad\|>'
    r'|</s>'
    r'|<s>'
    r'|\[INST\]'
    r'|\[/INST\]'
    r'|\[end\]',
  );

  /// Pattern that signals the model is hallucinating a new user turn — stop immediately.
  static final _userTurnPattern = RegExp(
    r'<\|user\|>|<\|im_start\|>\s*user|<start_of_turn>\s*user|\[INST\]',
  );

  /// Generate a streaming response.
  /// [messages] is a list of {role, content} maps.
  /// [systemPrompt] is prepended as a system message.
  /// Returns a Stream of String tokens.
  Stream<String> generate({
    required List<Map<String, String>> messages,
    String? systemPrompt,
    double temperature = 0.7,
  }) async* {
    if (ollamaActive.value) {
      yield* _generateOllama(
        messages: messages,
        systemPrompt: systemPrompt,
        temperature: temperature,
      );
      return;
    }

    // Prefer the model's own chat template (better answers for Qwen, Gemma,
    // Llama...). Falls back to the classic prompt if the template fails
    // before producing any text.
    if (_engine != null && isLoaded.value && !isGenerating.value) {
      var produced = false;
      try {
        final chatMessages = <LlamaChatMessage>[
          if (systemPrompt != null && systemPrompt.isNotEmpty)
            LlamaChatMessage.fromText(
                role: LlamaChatRole.system, text: systemPrompt),
          for (final m in messages)
            LlamaChatMessage.fromText(
              role: m['role'] == 'assistant'
                  ? LlamaChatRole.assistant
                  : m['role'] == 'system'
                      ? LlamaChatRole.system
                      : LlamaChatRole.user,
              text: m['content'] ?? '',
            ),
        ];
        final params = const GenerationParams(penalty: 1.1, topP: 0.95, minP: 0.05)
            .copyWith(temp: temperature);
        await for (final token in generateChatCompletion(
          messages: chatMessages,
          params: params,
        )) {
          produced = true;
          yield token;
        }
        return;
      } catch (e) {
        if (produced) rethrow;
        try {
          Get.find<LogService>().info(
              'Chat template failed, using classic prompt: $e', source: 'LLM');
        } catch (_) {}
      }
    }

    yield* _generateClassic(
      messages: messages,
      systemPrompt: systemPrompt,
      temperature: temperature,
    );
  }

  /// Classic generation with a generic prompt format (original behaviour).
  Stream<String> _generateClassic({
    required List<Map<String, String>> messages,
    String? systemPrompt,
    double temperature = 0.7,
  }) async* {
    if (_engine == null || !isLoaded.value) {
      throw StateError('No model loaded. Call loadModel() first.');
    }
    if (isGenerating.value) {
      throw StateError('Another generation is already in progress.');
    }

    isGenerating.value = true;
    tokensPerSecond.value = 0.0;
    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;

    // Buffer to detect multi-token stop sequences
    String buffer = '';

    try {
      // Build the full prompt from messages
      final prompt = _buildPrompt(messages, systemPrompt);

      await for (final token in _engine!.generate(prompt)) {
        tokenCount++;
        if (stopwatch.elapsedMilliseconds > 0) {
          tokensPerSecond.value =
              tokenCount / (stopwatch.elapsedMilliseconds / 1000);
        }

        // Accumulate into buffer for stop-pattern detection
        buffer += token;

        // Check if model is hallucinating a user turn — stop immediately
        if (_userTurnPattern.hasMatch(buffer)) {
          final cleaned = buffer
              .replaceAll(_stopPatterns, '')
              .replaceAll(_userTurnPattern, '')
              .trim();
          if (cleaned.isNotEmpty) {
            yield cleaned;
          }
          break;
        }

        // Check if buffer contains any stop pattern
        if (_stopPatterns.hasMatch(buffer)) {
          // Yield everything before the stop pattern, then stop
          final cleaned = buffer.replaceAll(_stopPatterns, '').trim();
          if (cleaned.isNotEmpty) {
            yield cleaned;
          }
          break;
        }

        // If buffer is getting long enough that we know it's safe, flush it
        // Keep last 30 chars to detect split stop sequences
        if (buffer.length > 40) {
          final safe = buffer.substring(0, buffer.length - 30);
          buffer = buffer.substring(buffer.length - 30);
          yield safe;
        }
      }

      // Flush any remaining buffer (cleaning all control patterns)
      if (buffer.isNotEmpty) {
        final cleaned = buffer
            .replaceAll(_stopPatterns, '')
            .replaceAll(_userTurnPattern, '')
            .trim();
        if (cleaned.isNotEmpty) {
          yield cleaned;
        }
      }
    } finally {
      stopwatch.stop();
      lastGenerationTokens.value = tokenCount;
      lastGenerationSpeed.value = tokensPerSecond.value;
      isGenerating.value = false;
    }
  }

  /// Generate a chat completion using llamadart's chat-template API.
  Stream<String> generateChatCompletion({
    required List<LlamaChatMessage> messages,
    GenerationParams params = const GenerationParams(),
  }) async* {
    if (_engine == null || !isLoaded.value) {
      throw StateError('No model loaded. Call loadModel() first.');
    }
    if (isGenerating.value) {
      throw StateError('Another generation is already in progress.');
    }

    isGenerating.value = true;
    tokensPerSecond.value = 0.0;
    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;

    try {
      await for (final chunk in _engine!.create(
        messages,
        params: params,
        toolChoice: ToolChoice.none,
      )) {
        final choice = chunk.choices.isNotEmpty ? chunk.choices.first : null;
        final content = choice?.delta.content;
        if (content == null || content.isEmpty) continue;

        tokenCount++;
        if (stopwatch.elapsedMilliseconds > 0) {
          tokensPerSecond.value =
              tokenCount / (stopwatch.elapsedMilliseconds / 1000);
        }
        yield content;
      }
    } finally {
      stopwatch.stop();
      lastGenerationTokens.value = tokenCount;
      lastGenerationSpeed.value = tokensPerSecond.value;
      isGenerating.value = false;
    }
  }

  Future<int> countTokens(String text) async {
    if (_engine == null || !isLoaded.value) return 0;
    try {
      return await _engine!.getTokenCount(text);
    } catch (_) {
      return 0;
    }
  }

  /// Stop ongoing generation.
  Future<void> stopGeneration() async {
    if (_ollamaClient != null) {
      _ollamaStopRequested = true;
      _ollamaClient!.close();
      _ollamaClient = null;
    }
    _generateSub?.cancel();
    _generateSub = null;
    _engine?.cancelGeneration();
    isGenerating.value = false;
  }

  /// Full native teardown — dispose engine AND backend to prevent stale state.
  Future<void> _fullTeardown() async {
    if (_engine != null) {
      try {
        await _engine!.dispose();
      } catch (_) {
        // Engine may already be in broken state — ignore
      }
      _engine = null;
    }
    // Also destroy the backend — it can't be reused after engine disposal
    _backend = null;
    ollamaActive.value = false;
    ollamaModel.value = '';
    isLoaded.value = false;
    loadedModelPath.value = '';
    tokensPerSecond.value = 0.0;
  }

  /// Unload the current model and free memory.
  Future<void> unloadModel() async {
    await _fullTeardown();

    // Disable wake lock when model is unloaded
    try {
      final wakelockService = Get.find<WakelockService>();
      await wakelockService.disable();
    } catch (_) {}
  }

  // ── Ollama ─────────────────────────────────────────────────

  static String _normalizeBaseUrl(String url) {
    var u = url.trim();
    if (u.isEmpty) u = 'http://127.0.0.1:11434';
    if (!u.startsWith('http://') && !u.startsWith('https://')) u = 'http://$u';
    while (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    if (u.endsWith('/v1')) u = u.substring(0, u.length - 3);
    return u;
  }

  /// Names of the models the Ollama server has (including `:cloud` ones).
  Future<List<String>> listOllamaModels(String baseUrl) async {
    final base = _normalizeBaseUrl(baseUrl);
    final resp = await http
        .get(Uri.parse('$base/api/tags'))
        .timeout(const Duration(seconds: 6));
    if (resp.statusCode != 200) {
      throw Exception('Ollama respondió ${resp.statusCode}');
    }
    final data = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    final models = (data['models'] as List? ?? const [])
        .map((m) => (m as Map)['name']?.toString() ?? '')
        .where((n) => n.isNotEmpty)
        .toList();
    return models;
  }

  /// Switch generation to an Ollama model. Unloads any local GGUF model.
  Future<void> connectOllama({required String baseUrl, required String model}) async {
    final base = _normalizeBaseUrl(baseUrl);
    final name = model.trim();
    if (name.isEmpty) throw Exception('Escribe el nombre del modelo de Ollama.');
    // Check the server is reachable before switching.
    try {
      await http.get(Uri.parse('$base/api/version')).timeout(const Duration(seconds: 5));
    } catch (_) {
      throw Exception(
          'No se puede conectar con Ollama en $base. ¿Está Ollama abierto?');
    }
    if (_engine != null || isLoaded.value) {
      await _fullTeardown();
    }
    _ollamaBaseUrl = base;
    ollamaModel.value = name;
    ollamaActive.value = true;
    loadedModelPath.value = 'ollama:$name';
    isLoaded.value = true;
    try {
      Get.find<LogService>().info('Connected to Ollama $base model=$name', source: 'LLM');
    } catch (_) {}
  }

  Future<void> disconnectOllama() async {
    await stopGeneration();
    ollamaActive.value = false;
    ollamaModel.value = '';
    if (_engine == null) {
      isLoaded.value = false;
      loadedModelPath.value = '';
    }
  }

  Stream<String> _generateOllama({
    required List<Map<String, String>> messages,
    String? systemPrompt,
    double temperature = 0.7,
  }) async* {
    if (isGenerating.value) {
      throw StateError('Another generation is already in progress.');
    }
    isGenerating.value = true;
    tokensPerSecond.value = 0.0;
    _ollamaStopRequested = false;
    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;
    final client = http.Client();
    _ollamaClient = client;

    try {
      final body = jsonEncode({
        'model': ollamaModel.value,
        'stream': true,
        'temperature': temperature,
        'messages': [
          if (systemPrompt != null && systemPrompt.isNotEmpty)
            {'role': 'system', 'content': systemPrompt},
          for (final m in messages)
            {'role': m['role'] ?? 'user', 'content': m['content'] ?? ''},
        ],
      });
      final req = http.Request('POST', Uri.parse('$_ollamaBaseUrl/v1/chat/completions'))
        ..headers['Content-Type'] = 'application/json'
        ..body = body;
      final resp = await client.send(req);
      if (resp.statusCode != 200) {
        final err = await resp.stream.bytesToString();
        if (resp.statusCode == 402) {
          throw Exception(
              'Ollama pide créditos o suscripción para este modelo (402). Revisa tu saldo en ollama.com.');
        }
        if (resp.statusCode == 404) {
          throw Exception(
              'Ollama no encuentra el modelo "${ollamaModel.value}". Ejecuta: ollama pull ${ollamaModel.value}');
        }
        throw Exception('Ollama error ${resp.statusCode}: $err');
      }

      await for (final line in resp.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())) {
        if (!line.startsWith('data:')) continue;
        final data = line.substring(5).trim();
        if (data == '[DONE]') break;
        try {
          final j = jsonDecode(data) as Map<String, dynamic>;
          final choices = j['choices'] as List?;
          if (choices == null || choices.isEmpty) continue;
          final delta = (choices.first as Map)['delta'] as Map?;
          final content = delta?['content'];
          if (content is String && content.isNotEmpty) {
            tokenCount++;
            if (stopwatch.elapsedMilliseconds > 0) {
              tokensPerSecond.value =
                  tokenCount / (stopwatch.elapsedMilliseconds / 1000);
            }
            yield content;
          }
        } catch (_) {
          // Ignore malformed keep-alive lines.
        }
      }
    } catch (e) {
      if (!_ollamaStopRequested) rethrow;
    } finally {
      stopwatch.stop();
      if (identical(_ollamaClient, client)) _ollamaClient = null;
      client.close();
      lastGenerationTokens.value = tokenCount;
      lastGenerationSpeed.value = tokensPerSecond.value;
      isGenerating.value = false;
    }
  }

  /// Build a single prompt string from chat messages.
  String _buildPrompt(
    List<Map<String, String>> messages,
    String? systemPrompt,
  ) {
    final buffer = StringBuffer();

    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      buffer.writeln('<|system|>');
      buffer.writeln(systemPrompt);
      buffer.writeln('<|end|>');
    }

    for (final msg in messages) {
      final role = msg['role'] ?? 'user';
      final content = msg['content'] ?? '';
      buffer.writeln('<|$role|>');
      buffer.writeln(content);
      buffer.writeln('<|end|>');
    }

    buffer.writeln('<|assistant|>');
    return buffer.toString();
  }

  @override
  void onClose() {
    unloadModel();
    super.onClose();
  }
}
