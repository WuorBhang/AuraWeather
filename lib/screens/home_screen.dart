import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'camera_screen.dart';
import 'results_screen.dart';
import '../services/ml_service.dart';
import '../widgets/weather_card.dart';

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const HomeScreen({super.key, required this.cameras});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    // Load the ML model when the app starts
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<MLService>().loadModel();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.blue.shade400,
              Colors.blue.shade600,
              Colors.blue.shade800,
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: _buildMainContent(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(20),
      child: const Column(
        children: [
          Icon(
            Icons.wb_cloudy,
            size: 60,
            color: Colors.white,
          ),
          SizedBox(height: 10),
          Text(
            'Weather AI',
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          Text(
            'Identify weather from photos',
            style: TextStyle(
              fontSize: 16,
              color: Colors.white70,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent() {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            Text(
              'How would you like to capture the weather?',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: Colors.grey.shade800,
              ),
            ),
            const SizedBox(height: 30),
            _buildCaptureOptions(),
            const SizedBox(height: 40),
            _buildMLStatus(),
            const Spacer(),
            _buildSampleWeatherCards(),
          ],
        ),
      ),
    );
  }

  Widget _buildCaptureOptions() {
    return Column(
      children: [
        _buildCaptureButton(
          icon: Icons.camera_alt,
          title: 'Take Photo',
          subtitle: 'Capture weather with camera',
          onTap: _openCamera,
          color: Colors.green,
        ),
        const SizedBox(height: 15),
        _buildCaptureButton(
          icon: Icons.photo_library,
          title: 'Choose from Gallery',
          subtitle: 'Select existing photo',
          onTap: _pickFromGallery,
          color: Colors.orange,
        ),
      ],
    );
  }

  Widget _buildCaptureButton({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    required Color color,
  }) {
    return SizedBox(
      width: double.infinity,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(15),
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(15),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(15),
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(icon, color: Colors.white, size: 24),
                ),
                const SizedBox(width: 20),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Colors.grey.shade800,
                        ),
                      ),
                      Text(
                        subtitle,
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.arrow_forward_ios,
                  color: color,
                  size: 18,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMLStatus() {
    return Consumer<MLService>(
      builder: (context, mlService, child) {
        return Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: mlService.isModelLoaded 
                ? Colors.green.shade50 
                : Colors.orange.shade50,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: mlService.isModelLoaded 
                  ? Colors.green.shade200 
                  : Colors.orange.shade200,
            ),
          ),
          child: Row(
            children: [
              Icon(
                mlService.isModelLoaded ? Icons.check_circle : Icons.hourglass_empty,
                color: mlService.isModelLoaded ? Colors.green : Colors.orange,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  mlService.isLoading 
                      ? 'Loading AI model...'
                      : mlService.isModelLoaded 
                          ? 'AI model ready for weather detection'
                          : 'AI model not loaded',
                  style: TextStyle(
                    color: mlService.isModelLoaded ? Colors.green.shade700 : Colors.orange.shade700,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildSampleWeatherCards() {
    final weatherTypes = ['Cloudy', 'Rain', 'Sunrise', 'Shine'];
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Weather Types We Can Detect:',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.grey.shade700,
          ),
        ),
        const SizedBox(height: 15),
        SizedBox(
          height: 80,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: weatherTypes.length,
            itemBuilder: (context, index) {
              return Container(
                margin: const EdgeInsets.only(right: 10),
                child: WeatherCard(
                  weatherType: weatherTypes[index],
                  isCompact: true,
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  void _openCamera() {
    if (widget.cameras.isEmpty) {
      _showError('No cameras available');
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CameraScreen(cameras: widget.cameras),
      ),
    );
  }

  void _pickFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (image != null) {
        _processImage(image.path);
      }
    } catch (e) {
      _showError('Error picking image: $e');
    }
  }

  void _processImage(String imagePath) async {
    // Show loading dialog
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(
        child: Card(
          child: Padding(
            padding: EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 15),
                Text('Analyzing weather...'),
              ],
            ),
          ),
        ),
      ),
    );

    try {
      final prediction = await context.read<MLService>().classifyImage(imagePath);
      
      Navigator.pop(context); // Close loading dialog
      
      if (prediction != null) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(prediction: prediction),
          ),
        );
      } else {
        _showError('Failed to analyze the image. Please try again.');
      }
    } catch (e) {
      Navigator.pop(context); // Close loading dialog
      _showError('Error analyzing image: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
      ),
    );
  }
}
