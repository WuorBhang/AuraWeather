import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import 'screens/home_screen.dart';
import 'services/ml_service.dart';
import 'services/ai_service.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
  }
  
  runApp(WeatherAIApp());
}

class WeatherAIApp extends StatelessWidget {
  const WeatherAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => MLService()),
        ChangeNotifierProvider(create: (_) => AIService()),
      ],
      child: MaterialApp(
        title: 'Weather AI',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          fontFamily: 'Roboto',
        ),
        home: HomeScreen(cameras: cameras),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
