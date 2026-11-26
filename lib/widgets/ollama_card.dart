import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../controllers/model_controller.dart';
import '../services/chat_storage_service.dart';
import '../services/llm_service.dart';
import '../services/tts_service.dart';
import '../theme/app_colors.dart';

/// Settings card to use Ollama models (local or `:cloud`) instead of a GGUF.
class OllamaCard extends StatefulWidget {
  const OllamaCard({super.key});

  @override
  State<OllamaCard> createState() => _OllamaCardState();
}

class _OllamaCardState extends State<OllamaCard> {
  final _llm = Get.find<LlmService>();
  final _storage = Get.find<ChatStorageService>();
  late final TextEditingController _urlCtrl;
  late final TextEditingController _modelCtrl;
  List<String> _models = [];
  bool _busy = false;
  String _status = '';
  bool _statusIsError = false;

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController(text: _storage.ollamaBaseUrl);
    _modelCtrl = TextEditingController(text: _storage.ollamaModel);
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    _modelCtrl.dispose();
    super.dispose();
  }

  void _setStatus(String msg, {bool error = false}) {
    if (!mounted) return;
    setState(() {
      _status = msg;
      _statusIsError = error;
    });
  }

  Future<void> _refreshModels() async {
    setState(() => _busy = true);
    try {
      final list = await _llm.listOllamaModels(_urlCtrl.text);
      setState(() => _models = list);
      _setStatus(list.isEmpty
          ? 'Ollama no tiene modelos. Ejecuta: ollama pull deepseek-v4.1-flash:cloud'
          : '${list.length} modelos encontrados.');
    } catch (e) {
      _setStatus('No se pudo conectar con Ollama. ¿Está abierto?', error: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _connect() async {
    setState(() => _busy = true);
    try {
      final model = _modelCtrl.text.trim();
      await _llm.connectOllama(baseUrl: _urlCtrl.text, model: model);
      _storage.ollamaBaseUrl = _llm.ollamaBaseUrl;
      _storage.ollamaModel = model;
      try {
        Get.find<ModelController>().selectedModelFilename.value = 'Ollama · $model';
      } catch (_) {}
      _setStatus('Conectado a $model. Ya puedes chatear.');
    } catch (e) {
      _setStatus(e.toString().replaceFirst('Exception: ', ''), error: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _disconnect() async {
    await _llm.disconnectOllama();
    try {
      final mc = Get.find<ModelController>();
      if ((mc.selectedModelFilename.value ?? '').startsWith('Ollama · ')) {
        mc.selectedModelFilename.value = null;
      }
    } catch (_) {}
    _setStatus('Desconectado. Carga un modelo local para volver a usarlo.');
  }

  InputDecoration _dec(BuildContext context, String label, String hint) {
    return InputDecoration(
      labelText: label,
      hintText: hint,
      isDense: true,
      labelStyle: TextStyle(color: context.textM, fontSize: 13),
      hintStyle: TextStyle(color: context.textD, fontSize: 13),
      filled: true,
      fillColor: context.bgInput,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: context.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: context.border),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.all(16),
      child: Obx(() {
        final active = _llm.ollamaActive.value;
        final current = _llm.ollamaModel.value;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.cloud_rounded,
                    color: active ? AppColors.green : context.textM, size: 20),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    active ? 'Usando Ollama: $current' : 'Ollama desconectado',
                    style: TextStyle(
                      color: context.text,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Usa los modelos de Ollama (por ejemplo los :cloud con tus créditos) '
              'en lugar de un modelo GGUF local. Ollama tiene que estar abierto.',
              style: TextStyle(color: context.textD, fontSize: 12, height: 1.4),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: _urlCtrl,
              style: TextStyle(color: context.text, fontSize: 14),
              decoration: _dec(context, 'Dirección de Ollama', 'http://127.0.0.1:11434'),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: _modelCtrl,
              style: TextStyle(color: context.text, fontSize: 14),
              decoration: _dec(context, 'Modelo', 'deepseek-v4.1-flash:cloud'),
            ),
            if (_models.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: _models
                    .map((m) => ActionChip(
                          label: Text(m, style: const TextStyle(fontSize: 12)),
                          onPressed: () => setState(() => _modelCtrl.text = m),
                        ))
                    .toList(),
              ),
            ],
            const SizedBox(height: 14),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  onPressed: _busy ? null : _connect,
                  icon: const Icon(Icons.link_rounded, size: 18),
                  label: Text(active ? 'Cambiar modelo' : 'Conectar'),
                  style: FilledButton.styleFrom(backgroundColor: AppColors.accent),
                ),
                OutlinedButton.icon(
                  onPressed: _busy ? null : _refreshModels,
                  icon: const Icon(Icons.refresh_rounded, size: 18),
                  label: const Text('Ver mis modelos'),
                ),
                if (active)
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _disconnect,
                    icon: const Icon(Icons.link_off_rounded, size: 18),
                    label: const Text('Desconectar'),
                  ),
              ],
            ),
            if (_busy) ...[
              const SizedBox(height: 12),
              const LinearProgressIndicator(minHeight: 2),
            ],
            if (_status.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                _status,
                style: TextStyle(
                  fontSize: 12,
                  color: _statusIsError ? AppColors.red : AppColors.green,
                ),
              ),
            ],
            const SizedBox(height: 8),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: Text('Conectar a Ollama al abrir la app',
                  style: TextStyle(color: context.text, fontSize: 13)),
              value: _storage.ollamaAutoConnect,
              onChanged: (v) => setState(() => _storage.ollamaAutoConnect = v),
              activeThumbColor: AppColors.accent,
            ),
          ],
        );
      }),
    );
  }
}

/// Settings card for voice options.
class VoiceCard extends StatefulWidget {
  const VoiceCard({super.key});

  @override
  State<VoiceCard> createState() => _VoiceCardState();
}

class _VoiceCardState extends State<VoiceCard> {
  final _storage = Get.find<ChatStorageService>();

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            secondary: Icon(Icons.record_voice_over_rounded, color: context.textM),
            title: Text('Leer las respuestas en voz alta',
                style: TextStyle(color: context.text, fontSize: 14)),
            subtitle: Text(
              'Cada respuesta nueva se lee sola. También puedes pulsar "Escuchar" en cualquier mensaje.',
              style: TextStyle(color: context.textD, fontSize: 12),
            ),
            value: _storage.autoSpeak,
            onChanged: (v) {
              setState(() => _storage.autoSpeak = v);
              if (!v) TtsService.instance.stop();
            },
            activeThumbColor: AppColors.accent,
          ),
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
              'Para hablarle: pulsa en la caja de texto y luego Win + H (dictado de Windows).',
              style: TextStyle(color: context.textD, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}
