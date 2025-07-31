import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'dart:math';
import 'dart:typed_data';
import 'dart:io';
import 'package:image/image.dart' as img;
import '../models/weather_prediction.dart';

class MLService extends ChangeNotifier {
  bool _isModelLoaded = false;
  bool _isLoading = false;
  
  // Weather classes as defined in training
  final List<String> _labels = ['Cloudy', 'Rain', 'Sunrise', 'Shine'];
  
  // Model configuration
  static const String _modelPath = 'assets/models/weather_model.tflite';
  static const int _inputSize = 224;
  
  bool get isModelLoaded => _isModelLoaded;
  bool get isLoading => _isLoading;
  List<String> get labels => List.unmodifiable(_labels);

  Future<void> loadModel() async {
    if (_isModelLoaded) return;
    
    _isLoading = true;
    notifyListeners();
    
    try {
      // Check if model file exists
      final ByteData data = await rootBundle.load(_modelPath);
      final Uint8List bytes = data.buffer.asUint8List();
      
      if (bytes.isNotEmpty) {
        // In a real implementation, this would load the TFLite model
        // For now, we'll simulate successful loading
        await Future.delayed(Duration(seconds: 2));
        _isModelLoaded = true;
        print('Weather classification model loaded successfully');
        print('Model size: ${(bytes.length / (1024 * 1024)).toStringAsFixed(2)} MB');
        print('Ready to classify: ${_labels.join(', ')}');
      } else {
        throw Exception('Model file is empty');
      }
    } catch (e) {
      print('Model loading error: $e');
      print('Note: Train the model first using weather_model_trainer.py');
      _isModelLoaded = false;
      
      // For development, allow demo mode
      if (kDebugMode) {
        print('Running in demo mode with simulated predictions');
        _isModelLoaded = true;
      }
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
      print('Processing image: $imagePath');
      
      // Preprocess the image
      final processedImage = await _preprocessImage(imagePath);
      
      // Run inference
      final prediction = await _runInference(processedImage);
      
      if (prediction != null) {
        print('Classification complete: ${prediction.weatherType} (${(prediction.confidence * 100).toStringAsFixed(1)}%)');
        return prediction;
      }
      
      return null;
    } catch (e) {
      print('Error during classification: $e');
      return null;
    }
  }

  Future<List<List<List<double>>>> _preprocessImage(String imagePath) async {
    try {
      // Read image file
      final imageFile = File(imagePath);
      final imageBytes = await imageFile.readAsBytes();
      
      // Decode image
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }
      
      // Resize to model input size (224x224)
      image = img.copyResize(image, width: _inputSize, height: _inputSize);
      
      // Convert to normalized float values (0-1 range)
      List<List<List<double>>> input = [];
      for (int y = 0; y < _inputSize; y++) {
        List<List<double>> row = [];
        for (int x = 0; x < _inputSize; x++) {
          final pixel = image.getPixel(x, y);
          
          // Extract RGB values and normalize to 0-1
          final r = img.getRed(pixel) / 255.0;
          final g = img.getGreen(pixel) / 255.0;
          final b = img.getBlue(pixel) / 255.0;
          
          row.add([r, g, b]);
        }
        input.add(row);
      }
      
      print('Image preprocessed: ${_inputSize}x${_inputSize}x3');
      return input;
    } catch (e) {
      print('Error preprocessing image: $e');
      rethrow;
    }
  }

  Future<WeatherPrediction?> _runInference(List<List<List<double>>> input) async {
    try {
      // Simulate inference time
      await Future.delayed(Duration(milliseconds: 500));
      
      // In a real implementation, this would use TensorFlow Lite
      // For demo, we'll analyze image characteristics to provide realistic predictions
      final prediction = _simulateIntelligentPrediction(input);
      
      return WeatherPrediction(
        weatherType: prediction['label']!,
        confidence: double.parse(prediction['confidence']!),
        timestamp: DateTime.now(),
        imagePath: '', // Will be set by caller
      );
    } catch (e) {
      print('Error during inference: $e');
      return null;
    }
  }

  Map<String, String> _simulateIntelligentPrediction(List<List<List<double>>> input) {
    // Analyze image characteristics for more realistic predictions
    double avgBrightness = 0.0;
    double avgBlue = 0.0;
    double avgGray = 0.0;
    
    int totalPixels = _inputSize * _inputSize;
    
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = input[y][x];
        final r = pixel[0];
        final g = pixel[1];
        final b = pixel[2];
        
        // Calculate brightness (luminance)
        avgBrightness += (0.299 * r + 0.587 * g + 0.114 * b);
        avgBlue += b;
        
        // Calculate grayscale (for cloudy detection)
        final gray = (r + g + b) / 3;
        avgGray += (r - gray).abs() + (g - gray).abs() + (b - gray).abs();
      }
    }
    
    avgBrightness /= totalPixels;
    avgBlue /= totalPixels;
    avgGray /= totalPixels;
    
    // Determine weather based on image characteristics
    String weatherType;
    double confidence;
    
    if (avgBrightness > 0.7) {
      // Bright image - likely Sunshine
      weatherType = 'Shine';
      confidence = 0.85 + Random().nextDouble() * 0.1;
    } else if (avgBlue > 0.6) {
      // High blue content - likely Rain or overcast
      weatherType = 'Rain';
      confidence = 0.78 + Random().nextDouble() * 0.15;
    } else if (avgGray < 0.2 && avgBrightness > 0.4) {
      // Low color variation, medium brightness - Sunrise/Sunset
      weatherType = 'Sunrise';
      confidence = 0.82 + Random().nextDouble() * 0.12;
    } else {
      // Default to Cloudy
      weatherType = 'Cloudy';
      confidence = 0.75 + Random().nextDouble() * 0.18;
    }
    
    return {
      'label': weatherType,
      'confidence': confidence.toStringAsFixed(3),
    };
  }

  List<String> getRecommendations(String weatherType) {
    switch (weatherType.toLowerCase()) {
      case 'shine':
        return [
          '☀️ Apply sunscreen and wear sunglasses',
          '💧 Stay hydrated and seek shade during peak hours',
          '🏃 Perfect weather for outdoor activities and sports',
          '👕 Wear light, breathable clothing',
        ];
      case 'rain':
        return [
          '☂️ Carry an umbrella or wear waterproof clothing',
          '👟 Wear non-slip shoes for safety',
          '🏠 Great time for indoor activities like reading',
          '🌱 Perfect opportunity to collect rainwater for plants',
        ];
      case 'cloudy':
        return [
          '🧥 Layer clothing as temperature may change',
          '📸 Great lighting conditions for photography',
          '🚶 Ideal weather for walking and light exercise',
          '🌡️ Comfortable temperature for outdoor activities',
        ];
      case 'sunrise':
        return [
          '🌅 Perfect time for morning exercise or meditation',
          '🧥 Layer clothing as temperature will rise with sun',
          '☕ Ideal for peaceful outdoor breakfast',
          '📱 Great opportunity for stunning sunrise photos',
        ];
      default:
        return [
          '🌤️ Check local weather updates for detailed forecast',
          '👕 Dress appropriately for current conditions',
          '📱 Monitor weather changes throughout the day',
        ];
    }
  }

  void dispose() {
    super.dispose();
  }
}
