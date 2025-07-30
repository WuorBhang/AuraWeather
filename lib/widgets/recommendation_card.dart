import 'package:flutter/material.dart';
import '../models/weather_prediction.dart';

class RecommendationCard extends StatefulWidget {
  final AIRecommendation recommendation;
  final Color weatherColor;

  const RecommendationCard({
    Key? key,
    required this.recommendation,
    required this.weatherColor,
  }) : super(key: key);

  @override
  _RecommendationCardState createState() => _RecommendationCardState();
}

class _RecommendationCardState extends State<RecommendationCard>
    with SingleTickerProviderStateMixin {
  bool _isExpanded = false;
  late AnimationController _animationController;
  late Animation<double> _expandAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: Duration(milliseconds: 300),
      vsync: this,
    );
    _expandAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.only(bottom: 15),
      child: Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(15),
        ),
        child: Column(
          children: [
            _buildHeader(),
            AnimatedBuilder(
              animation: _expandAnimation,
              builder: (context, child) {
                return ClipRect(
                  child: Align(
                    alignment: Alignment.topCenter,
                    heightFactor: _expandAnimation.value,
                    child: child,
                  ),
                );
              },
              child: _buildExpandedContent(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return InkWell(
      onTap: _toggleExpanded,
      borderRadius: BorderRadius.vertical(
        top: Radius.circular(15),
        bottom: _isExpanded ? Radius.zero : Radius.circular(15),
      ),
      child: Container(
        padding: EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: widget.weatherColor.withOpacity(0.05),
          borderRadius: BorderRadius.vertical(
            top: Radius.circular(15),
            bottom: _isExpanded ? Radius.zero : Radius.circular(15),
          ),
        ),
        child: Row(
          children: [
            Container(
              padding: EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: widget.weatherColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                _getCategoryIcon(widget.recommendation.category),
                color: widget.weatherColor,
                size: 20,
              ),
            ),
            SizedBox(width: 15),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    widget.recommendation.title,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey.shade800,
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    widget.recommendation.description,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey.shade600,
                    ),
                  ),
                ],
              ),
            ),
            AnimatedRotation(
              turns: _isExpanded ? 0.5 : 0,
              duration: Duration(milliseconds: 300),
              child: Icon(
                Icons.expand_more,
                color: widget.weatherColor,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExpandedContent() {
    if (widget.recommendation.tips.isEmpty) {
      return Container(
        padding: EdgeInsets.all(20),
        child: Text(
          'No specific tips available for this recommendation.',
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey.shade600,
            fontStyle: FontStyle.italic,
          ),
        ),
      );
    }

    return Container(
      padding: EdgeInsets.fromLTRB(20, 0, 20, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Divider(color: Colors.grey.shade200),
          SizedBox(height: 10),
          Text(
            'Tips & Recommendations:',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey.shade700,
            ),
          ),
          SizedBox(height: 10),
          ...widget.recommendation.tips.map((tip) => _buildTipItem(tip)).toList(),
        ],
      ),
    );
  }

  Widget _buildTipItem(String tip) {
    return Container(
      margin: EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: EdgeInsets.only(top: 6),
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: widget.weatherColor,
              borderRadius: BorderRadius.circular(3),
            ),
          ),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              tip,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade700,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  IconData _getCategoryIcon(String category) {
    switch (category.toLowerCase()) {
      case 'clothing':
        return Icons.checkroom;
      case 'activities':
        return Icons.directions_run;
      case 'safety':
        return Icons.security;
      case 'health':
        return Icons.favorite;
      case 'general':
      default:
        return Icons.lightbulb_outline;
    }
  }

  void _toggleExpanded() {
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _animationController.forward();
      } else {
        _animationController.reverse();
      }
    });
  }
}
