import 'package:flutter/material.dart';

class WeatherPrediction {
  final String weatherType;
  final double confidence;
  final DateTime timestamp;
  final String imagePath;

  WeatherPrediction({
    required this.weatherType,
    required this.confidence,
    required this.timestamp,
    required this.imagePath,
  });

  Map<String, dynamic> toJson() {
    return {
      'weatherType': weatherType,
      'confidence': confidence,
      'timestamp': timestamp.toIso8601String(),
      'imagePath': imagePath,
    };
  }

  factory WeatherPrediction.fromJson(Map<String, dynamic> json) {
    return WeatherPrediction(
      weatherType: json['weatherType'],
      confidence: json['confidence'],
      timestamp: DateTime.parse(json['timestamp']),
      imagePath: json['imagePath'],
    );
  }

  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
  
  Color get weatherColor {
    switch (weatherType.toLowerCase()) {
      case 'cloudy':
        return Colors.grey.shade600;
      case 'rain':
        return Colors.blue.shade700;
      case 'sunrise':
        return Colors.orange.shade600;
      case 'shine':
        return Colors.yellow.shade600;
      default:
        return Colors.grey;
    }
  }
  
  IconData get weatherIcon {
    switch (weatherType.toLowerCase()) {
      case 'cloudy':
        return Icons.cloud;
      case 'rain':
        return Icons.grain;
      case 'sunrise':
        return Icons.wb_sunny;
      case 'shine':
        return Icons.wb_sunny;
      default:
        return Icons.help;
    }
  }
}

class AIRecommendation {
  final String title;
  final String description;
  final List<String> tips;
  final String category;

  AIRecommendation({
    required this.title,
    required this.description,
    required this.tips,
    required this.category,
  });

  factory AIRecommendation.fromJson(Map<String, dynamic> json) {
    return AIRecommendation(
      title: json['title'] ?? '',
      description: json['description'] ?? '',
      tips: List<String>.from(json['tips'] ?? []),
      category: json['category'] ?? '',
    );
  }
}
