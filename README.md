# weather_ai_app

# Weather AI - Flutter Mobile App

A Flutter mobile application that uses deep learning to classify weather conditions from images and provides AI-powered recommendations.

## Features

### 🤖 AI-Powered Weather Classification
- **4 Weather Types**: Cloudy, Rain, Sunrise, Shine
- **Deep Learning**: EfficientNetB0 with transfer learning
- **Real-time Processing**: TensorFlow Lite for mobile inference
- **High Accuracy**: Transfer learning on ImageNet with custom weather classification head

### 📱 Mobile App Features
- **Camera Integration**: Capture weather photos instantly
- **Gallery Picker**: Select existing images for analysis
- **Live Classification**: Real-time weather detection with confidence scores
- **Smart Recommendations**: AI-generated suggestions based on weather conditions
- **Professional UI**: Gradient design with mobile-first experience

### 🧠 Machine Learning Pipeline
- **Training**: Complete Python pipeline with weather_model_trainer.py
- **Model Architecture**: EfficientNetB0 + custom classification head
- **Optimization**: Automated TensorFlow Lite conversion for mobile deployment
- **Data Handling**: Image preprocessing, augmentation, and normalization

## Quick Start

### 1. Add Your Weather Images
Place your weather training images in the following structure:
```
weather_images/
├── train_dir/
│   ├── Cloudy/     # Your cloudy weather images
│   ├── Rain/       # Your rainy weather images  
│   ├── Sunrise/    # Your sunrise/sunset images
│   └── Shine/      # Your sunny weather images
└── test_dir/       # Optional test images with same structure
```

### 2. Train the AI Model
```bash
# Check your data structure
python training_demo.py

# Train the weather classification model
python weather_model_trainer.py
```

### 3. Run the Flutter App
```bash
# Install dependencies
flutter pub get

# Run the app
flutter run
```

## Project Architecture

### Training Pipeline (`weather_model_trainer.py`)
- **Phase 1**: Transfer learning with frozen EfficientNetB0 base
- **Phase 2**: Fine-tuning with unfrozen layers
- **Output**: Optimized weather_model.tflite for mobile deployment
- **Evaluation**: Accuracy metrics, confusion matrix, classification reports

### Flutter Integration (`lib/services/ml_service.dart`)
- **Model Loading**: Automatic TensorFlow Lite model initialization
- **Image Processing**: 224x224 resize and normalization
- **Inference**: Real-time weather classification
- **Recommendations**: Context-aware suggestions for each weather type

### UI Components
- **Home Screen**: Weather capture options and model status
- **Camera Screen**: Live camera integration with capture functionality
- **Results Screen**: Classification results with confidence and recommendations

## Model Performance

### Training Configuration
- **Input Size**: 224x224x3 RGB images
- **Base Model**: EfficientNetB0 (ImageNet pretrained)
- **Classes**: 4 weather conditions
- **Optimization**: TensorFlow Lite with DEFAULT optimizations
- **Mobile Ready**: Optimized for on-device inference

### Expected Accuracy
- **Transfer Learning**: High accuracy through pretrained weights
- **Data Augmentation**: Rotation, zoom, brightness variations
- **Validation**: 20% holdout for model evaluation
- **Performance**: Real-time classification on mobile devices

## Development Features

### Current Demo
- **Web Interface**: Complete weather classification demo at `/web_demo.html`
- **Interactive UI**: Test camera and gallery functionality
- **Smart Analysis**: Intelligent predictions based on image characteristics
- **Full Pipeline**: End-to-end weather classification with recommendations

### Ready for Production
- **TensorFlow Lite**: Mobile-optimized model format
- **Flutter Integration**: Complete MLService with preprocessing
- **Error Handling**: Robust image loading and inference pipeline
- **User Experience**: Professional interface with confidence indicators

## Getting Started

1. **Clone and Setup**
   ```bash
   flutter pub get
   ```

2. **Add Training Data**
   - Place weather images in `weather_images/train_dir/[class]/`
   - Ensure balanced dataset across all 4 weather types

3. **Train Model**
   ```bash
   python weather_model_trainer.py
   ```

4. **Test App**
   ```bash
   flutter run
   ```

## Technical Stack

- **Frontend**: Flutter (Dart)
- **ML Framework**: TensorFlow/Keras → TensorFlow Lite
- **Computer Vision**: Image preprocessing and classification
- **Architecture**: EfficientNetB0 with transfer learning
- **Deployment**: Mobile-optimized inference

## Project Status

✅ Complete Flutter app structure with camera integration  
✅ Comprehensive training pipeline with EfficientNetB0  
✅ TensorFlow Lite conversion and mobile optimization  
✅ Smart recommendations system with context awareness  
✅ Professional UI with gradient design and mobile UX  
✅ Ready for training with your weather image dataset  

**Next Step**: Add your weather images and run the training pipeline!

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
