import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import '../models/weather_prediction.dart';

class AIService extends ChangeNotifier {
  bool _isLoading = false;
  String? _lastError;
  
  bool get isLoading => _isLoading;
  String? get lastError => _lastError;

  Future<List<AIRecommendation>> getWeatherRecommendations(
    WeatherPrediction prediction,
  ) async {
    _isLoading = true;
    _lastError = null;
    notifyListeners();

    try {
      final apiKey = Platform.environment['OPENAI_API_KEY'] ?? 'sk-your-openai-key-here';
      
      final prompt = _buildPrompt(prediction);
      
      final response = await http.post(
        Uri.parse('https://api.openai.com/v1/chat/completions'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: json.encode({
          'model': 'gpt-3.5-turbo',
          'messages': [
            {
              'role': 'system',
              'content': 'You are a helpful weather assistant that provides practical advice and recommendations based on weather conditions. Always respond in JSON format with structured recommendations.'
            },
            {
              'role': 'user',
              'content': prompt,
            }
          ],
          'max_tokens': 800,
          'temperature': 0.7,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final content = data['choices'][0]['message']['content'];
        
        // Parse the AI response and convert to recommendations
        return _parseRecommendations(content, prediction.weatherType);
      } else {
        _lastError = 'Failed to get AI recommendations: ${response.statusCode}';
        return _getFallbackRecommendations(prediction.weatherType);
      }
    } catch (e) {
      _lastError = 'Error getting AI recommendations: $e';
      print('AI Service Error: $e');
      return _getFallbackRecommendations(prediction.weatherType);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  String _buildPrompt(WeatherPrediction prediction) {
    return '''
Based on the detected weather condition "${prediction.weatherType}" with ${prediction.confidencePercentage} confidence, 
provide practical recommendations in JSON format with the following structure:

{
  "recommendations": [
    {
      "title": "Clothing Advice",
      "description": "What to wear for this weather",
      "tips": ["tip1", "tip2", "tip3"],
      "category": "clothing"
    },
    {
      "title": "Activity Suggestions",
      "description": "Best activities for this weather",
      "tips": ["tip1", "tip2", "tip3"],
      "category": "activities"
    },
    {
      "title": "Safety & Health",
      "description": "Health and safety considerations",
      "tips": ["tip1", "tip2", "tip3"],
      "category": "safety"
    }
  ]
}

Make the recommendations specific, practical, and actionable for the weather condition: ${prediction.weatherType}.
''';
  }

  List<AIRecommendation> _parseRecommendations(String content, String weatherType) {
    try {
      // Try to extract JSON from the response
      final jsonStart = content.indexOf('{');
      final jsonEnd = content.lastIndexOf('}') + 1;
      
      if (jsonStart != -1 && jsonEnd > jsonStart) {
        final jsonContent = content.substring(jsonStart, jsonEnd);
        final parsed = json.decode(jsonContent);
        
        if (parsed['recommendations'] != null) {
          return (parsed['recommendations'] as List)
              .map((rec) => AIRecommendation.fromJson(rec))
              .toList();
        }
      }
      
      return _getFallbackRecommendations(weatherType);
    } catch (e) {
      print('Error parsing AI recommendations: $e');
      return _getFallbackRecommendations(weatherType);
    }
  }

  List<AIRecommendation> _getFallbackRecommendations(String weatherType) {
    switch (weatherType.toLowerCase()) {
      case 'cloudy':
        return [
          AIRecommendation(
            title: 'Cloudy Day Essentials',
            description: 'Perfect weather for outdoor activities without harsh sun',
            tips: [
              'Light layers are ideal - you can add or remove as needed',
              'Great day for photography with soft, diffused lighting',
              'Perfect weather for hiking or outdoor sports',
              'No need for heavy sun protection, but keep sunglasses handy'
            ],
            category: 'general',
          ),
        ];
      case 'rain':
        return [
          AIRecommendation(
            title: 'Rainy Day Preparation',
            description: 'Stay dry and make the most of wet weather',
            tips: [
              'Carry an umbrella or wear a waterproof jacket',
              'Wear non-slip shoes to avoid accidents',
              'Perfect time for indoor activities like reading or cooking',
              'Check for flood warnings in your area',
              'Great opportunity to collect rainwater for plants'
            ],
            category: 'general',
          ),
        ];
      case 'sunrise':
        return [
          AIRecommendation(
            title: 'Beautiful Sunrise Moments',
            description: 'Make the most of this magical time of day',
            tips: [
              'Perfect time for outdoor exercise or meditation',
              'Great lighting for photography - golden hour magic',
              'Layer clothing as temperature may change quickly',
              'Ideal time for a peaceful walk or jog',
              'Consider having breakfast outdoors'
            ],
            category: 'general',
          ),
        ];
      case 'shine':
        return [
          AIRecommendation(
            title: 'Sunny Day Guidelines',
            description: 'Enjoy the sunshine safely and comfortably',
            tips: [
              'Apply sunscreen 30 minutes before going outside',
              'Wear sunglasses and a hat for UV protection',
              'Stay hydrated - drink water regularly',
              'Perfect weather for outdoor activities and sports',
              'Seek shade during peak sun hours (10am-4pm)',
              'Light, breathable clothing recommended'
            ],
            category: 'general',
          ),
        ];
      default:
        return [
          AIRecommendation(
            title: 'General Weather Tips',
            description: 'Stay prepared for changing weather conditions',
            tips: [
              'Check weather updates regularly',
              'Dress in layers for temperature changes',
              'Keep emergency supplies handy',
              'Stay informed about local weather conditions'
            ],
            category: 'general',
          ),
        ];
    }
  }
}
