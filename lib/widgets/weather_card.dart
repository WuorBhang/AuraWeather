import 'package:flutter/material.dart';

class WeatherCard extends StatelessWidget {
  final String weatherType;
  final bool isCompact;
  final VoidCallback? onTap;

  const WeatherCard({
    super.key,
    required this.weatherType,
    this.isCompact = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = _getWeatherColor(weatherType);
    final icon = _getWeatherIcon(weatherType);
    
    if (isCompact) {
      return _buildCompactCard(color, icon);
    }
    
    return _buildFullCard(color, icon);
  }

  Widget _buildCompactCard(Color color, IconData icon) {
    return Container(
      width: 100,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            icon,
            color: color,
            size: 28,
          ),
          const SizedBox(height: 8),
          Text(
            weatherType,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: color,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildFullCard(Color color, IconData icon) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                color.withOpacity(0.1),
                color.withOpacity(0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(15),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                color: color,
                size: 48,
              ),
              const SizedBox(height: 15),
              Text(
                weatherType,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _getWeatherDescription(weatherType),
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade600,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _getWeatherColor(String weatherType) {
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

  IconData _getWeatherIcon(String weatherType) {
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

  String _getWeatherDescription(String weatherType) {
    switch (weatherType.toLowerCase()) {
      case 'cloudy':
        return 'Overcast skies with clouds';
      case 'rain':
        return 'Precipitation and wet conditions';
      case 'sunrise':
        return 'Beautiful dawn lighting';
      case 'shine':
        return 'Bright sunny conditions';
      default:
        return 'Weather condition';
    }
  }
}
