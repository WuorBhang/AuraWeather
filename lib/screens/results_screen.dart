import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import '../models/weather_prediction.dart';
import '../services/ai_service.dart';
import '../widgets/weather_card.dart';
import '../widgets/recommendation_card.dart';

class ResultsScreen extends StatefulWidget {
  final WeatherPrediction prediction;

  const ResultsScreen({Key? key, required this.prediction}) : super(key: key);

  @override
  _ResultsScreenState createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  List<AIRecommendation> _recommendations = [];
  bool _hasLoadedRecommendations = false;

  @override
  void initState() {
    super.initState();
    _loadRecommendations();
  }

  Future<void> _loadRecommendations() async {
    if (_hasLoadedRecommendations) return;
    
    final aiService = context.read<AIService>();
    final recommendations = await aiService.getWeatherRecommendations(widget.prediction);
    
    if (mounted) {
      setState(() {
        _recommendations = recommendations;
        _hasLoadedRecommendations = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: widget.prediction.weatherColor.withOpacity(0.1),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: _buildContent(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            widget.prediction.weatherColor,
            widget.prediction.weatherColor.withOpacity(0.8),
          ],
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: Icon(
                  Icons.arrow_back,
                  color: Colors.white,
                  size: 28,
                ),
              ),
              Spacer(),
              Text(
                'Weather Analysis',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
              Spacer(),
              IconButton(
                onPressed: _shareResults,
                icon: Icon(
                  Icons.share,
                  color: Colors.white,
                  size: 24,
                ),
              ),
            ],
          ),
          SizedBox(height: 20),
          Icon(
            widget.prediction.weatherIcon,
            size: 60,
            color: Colors.white,
          ),
          SizedBox(height: 15),
          Text(
            widget.prediction.weatherType,
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          Text(
            '${widget.prediction.confidencePercentage} confidence',
            style: TextStyle(
              fontSize: 16,
              color: Colors.white.withOpacity(0.9),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildContent() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
      ),
      child: Column(
        children: [
          _buildImageSection(),
          Expanded(
            child: _buildRecommendationsSection(),
          ),
          _buildBottomActions(),
        ],
      ),
    );
  }

  Widget _buildImageSection() {
    return Container(
      padding: EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(height: 10),
          Text(
            'Analyzed Image',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Colors.grey.shade800,
            ),
          ),
          SizedBox(height: 15),
          Center(
            child: Container(
              width: double.infinity,
              height: 200,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(15),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: Image.file(
                  File(widget.prediction.imagePath),
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      color: Colors.grey.shade200,
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.broken_image,
                              size: 50,
                              color: Colors.grey.shade400,
                            ),
                            SizedBox(height: 10),
                            Text(
                              'Image not available',
                              style: TextStyle(
                                color: Colors.grey.shade600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
          SizedBox(height: 15),
          _buildConfidenceBar(),
        ],
      ),
    );
  }

  Widget _buildConfidenceBar() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Confidence Level',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Colors.grey.shade700,
              ),
            ),
            Text(
              widget.prediction.confidencePercentage,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: widget.prediction.weatherColor,
              ),
            ),
          ],
        ),
        SizedBox(height: 8),
        Container(
          height: 8,
          decoration: BoxDecoration(
            color: Colors.grey.shade200,
            borderRadius: BorderRadius.circular(4),
          ),
          child: FractionallySizedBox(
            alignment: Alignment.centerLeft,
            widthFactor: widget.prediction.confidence,
            child: Container(
              decoration: BoxDecoration(
                color: widget.prediction.weatherColor,
                borderRadius: BorderRadius.circular(4),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRecommendationsSection() {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'AI Recommendations',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Colors.grey.shade800,
            ),
          ),
          SizedBox(height: 15),
          Expanded(
            child: Consumer<AIService>(
              builder: (context, aiService, child) {
                if (aiService.isLoading && !_hasLoadedRecommendations) {
                  return _buildLoadingRecommendations();
                }
                
                if (_recommendations.isEmpty) {
                  return _buildEmptyRecommendations(aiService.lastError);
                }
                
                return ListView.builder(
                  itemCount: _recommendations.length,
                  itemBuilder: (context, index) {
                    return RecommendationCard(
                      recommendation: _recommendations[index],
                      weatherColor: widget.prediction.weatherColor,
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingRecommendations() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(
            color: widget.prediction.weatherColor,
          ),
          SizedBox(height: 20),
          Text(
            'Getting AI recommendations...',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey.shade600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyRecommendations(String? error) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.psychology,
            size: 60,
            color: Colors.grey.shade400,
          ),
          SizedBox(height: 20),
          Text(
            error != null ? 'Failed to load recommendations' : 'No recommendations available',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w500,
              color: Colors.grey.shade600,
            ),
          ),
          if (error != null) ...[
            SizedBox(height: 10),
            Text(
              error,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade500,
              ),
              textAlign: TextAlign.center,
            ),
          ],
          SizedBox(height: 20),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _hasLoadedRecommendations = false;
              });
              _loadRecommendations();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: widget.prediction.weatherColor,
            ),
            child: Text('Retry', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomActions() {
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        border: Border(top: BorderSide(color: Colors.grey.shade200)),
      ),
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton(
              onPressed: _analyzeAnother,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: widget.prediction.weatherColor,
                side: BorderSide(color: widget.prediction.weatherColor),
                padding: EdgeInsets.symmetric(vertical: 15),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text(
                'Analyze Another',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SizedBox(width: 15),
          Expanded(
            child: ElevatedButton(
              onPressed: _saveResults,
              style: ElevatedButton.styleFrom(
                backgroundColor: widget.prediction.weatherColor,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 15),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text(
                'Save Results',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _shareResults() {
    // Implement sharing functionality
    final message = '''
Weather Analysis Results:
🌤️ Detected: ${widget.prediction.weatherType}
📊 Confidence: ${widget.prediction.confidencePercentage}
📅 Date: ${widget.prediction.timestamp.toString().split(' ')[0]}

Analyzed with Weather AI app
''';

    print('Share: $message'); // Placeholder for actual sharing
    
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Sharing functionality would be implemented here'),
        backgroundColor: widget.prediction.weatherColor,
      ),
    );
  }

  void _analyzeAnother() {
    Navigator.popUntil(context, (route) => route.isFirst);
  }

  void _saveResults() {
    // Implement save functionality
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Results saved successfully!'),
        backgroundColor: Colors.green,
        duration: Duration(seconds: 2),
      ),
    );
  }
}
