import 'package:flutter_tts/flutter_tts.dart';
import 'package:get/get.dart';

/// Reads AI answers aloud with the system's text-to-speech voices.
class TtsService {
  TtsService._();
  static final TtsService instance = TtsService._();

  FlutterTts? _tts;
  bool _ready = false;

  /// Id of the message being read right now (null when silent).
  final speakingId = RxnString();

  Future<void> _init() async {
    if (_ready) return;
    final tts = FlutterTts();
    try {
      await tts.setLanguage('es-ES');
    } catch (_) {}
    try {
      await tts.awaitSpeakCompletion(false);
    } catch (_) {}
    tts.setCompletionHandler(() => speakingId.value = null);
    tts.setCancelHandler(() => speakingId.value = null);
    tts.setErrorHandler((_) => speakingId.value = null);
    _tts = tts;
    _ready = true;
  }

  /// Strip markdown so the voice doesn't read symbols aloud.
  static String clean(String text) {
    var t = text;
    t = t.replaceAll(RegExp(r'```[\s\S]*?```'), ' (bloque de código) ');
    t = t.replaceAll(RegExp(r'`([^`]*)`'), r'$1');
    t = t.replaceAll(RegExp(r'!\[[^\]]*\]\([^)]*\)'), '');
    t = t.replaceAllMapped(RegExp(r'\[([^\]]+)\]\([^)]*\)'), (m) => m[1] ?? '');
    t = t.replaceAll(RegExp(r'^\s{0,3}#{1,6}\s*', multiLine: true), '');
    t = t.replaceAll(RegExp(r'^\s*[-*+]\s+', multiLine: true), '');
    t = t.replaceAll(RegExp(r'[*_~>|]'), '');
    t = t.replaceAll(RegExp(r'\n{2,}'), '. ');
    return t.trim();
  }

  Future<void> speak(String id, String text) async {
    final cleaned = clean(text);
    if (cleaned.isEmpty) return;
    await _init();
    await _tts!.stop();
    speakingId.value = id;
    await _tts!.speak(cleaned);
  }

  Future<void> stop() async {
    if (_tts != null) {
      await _tts!.stop();
    }
    speakingId.value = null;
  }

  Future<void> toggle(String id, String text) async {
    if (speakingId.value == id) {
      await stop();
    } else {
      await speak(id, text);
    }
  }
}
