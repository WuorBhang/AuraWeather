import 'package:flutter/foundation.dart';
import 'dart:math';
import '../models/weather_prediction.dart';

class MLService extends ChangeNotifier {
  bool _isModelLoaded = false;
  bool _isLoading = false;
  
  final List<String> _labels = ['Cloudy', 'Rain', 'Sunrise', 'Shine'];
  
  bool get isModelLoaded => _isModelLoaded;
  bool get isLoading => _isLoading;

  Future<void> loadModel() async {
    if (_isModelLoaded) return;
    
    _isLoading = true;
    notifyListeners();
    
    try {
      // Simulate model loading for web demo
      await Future.delayed(Duration(seconds: 2));
      
      _isModelLoaded = true;
      print('Weather classification model loaded successfully (Demo Mode)');
    } catch (e) {
      print('Error loading model: $e');
      _isModelLoaded = false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<WeatherPrediction?> classifyImage(String imagePath) async {
    if (!_isModelLoaded) {
      await loadModel();
      if (!_isModelLoaded) return null;
    }

    try {
      // Simulate inference delay
      await Future.delayed(Duration(seconds: 1));
      
      // Generate random prediction for demo
      final random = Random();
      final randomIndex = random.nextInt(_labels.length);
      final confidence = 0.7 + (random.nextDouble() * 0.25); // 70-95% confidence
      
      return WeatherPrediction(
        weatherType: _labels[randomIndex],
        confidence: confidence,
        timestamp: DateTime.now(),
        imagePath: imagePath,
      );
    } catch (e) {
      print('Error during classification: $e');
      return null;
    }
  }

  void dispose() {
    super.dispose();
  }
}
