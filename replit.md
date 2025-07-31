# Flutter Weather AI Image Classifier

## Overview

This project is a Flutter mobile application that uses machine learning to classify weather conditions from user-uploaded images. The system uses a TensorFlow model trained on weather images (Cloudy, Rain, Sunrise, Shine) and converts it to TensorFlow Lite format for mobile deployment.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The project follows a hybrid architecture combining machine learning model training with mobile application deployment:

### Training Phase
- **Framework**: TensorFlow/Keras for model training
- **Model Architecture**: EfficientNetB0 pre-trained on ImageNet with custom classification head
- **Data Pipeline**: TensorFlow Datasets (TFDS) for data loading and preprocessing
- **Image Processing**: Standardized 224x224 pixel input size with data augmentation

### Mobile Deployment Phase
- **Platform**: Flutter for cross-platform mobile development
- **Model Format**: TensorFlow Lite (.tflite) for optimized mobile inference
- **Converter**: Python script to convert trained models to mobile-ready format

## Key Components

### 1. Model Architecture (`model_converter.py`)
- **Base Model**: EfficientNetB0 with frozen pre-trained weights
- **Classification Head**: Global average pooling + batch normalization + dropout (0.2) + dense layer
- **Output**: 4-class softmax for weather conditions (Cloudy, Rain, Sunrise, Shine)
- **Input Shape**: (224, 224, 3) RGB images

### 2. Model Conversion Pipeline
- **Purpose**: Convert TensorFlow models to TensorFlow Lite format for mobile deployment
- **Flexibility**: Handles both SavedModel directories and .h5/.keras files
- **Output**: Optimized .tflite model for Flutter integration

### 3. Data Structure
- **Training Data**: `train_dir/` containing subdirectories for each weather class
- **Testing Data**: `test_dir/` with same structure as training data
- **Classes**: 4 weather conditions (Cloudy, Rain, Sunrise, Shine)
- **Preprocessing**: Image resizing to 224x224, data augmentation for training

## Data Flow

1. **Training Phase**:
   - Raw images → TFDS loading → Image preprocessing (resize, augmentation)
   - Preprocessed data → EfficientNetB0 model → Training with frozen base + trainable head
   - Trained model → Save as .h5/.keras or SavedModel format

2. **Conversion Phase**:
   - Saved TensorFlow model → TFLite converter → Optimized .tflite model

3. **Mobile Inference** (planned):
   - User image upload → Image preprocessing → TFLite model inference → Weather classification result

## External Dependencies

### Python/ML Dependencies
- **TensorFlow**: Core ML framework for model training and conversion
- **Keras**: High-level API for model building
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization during training
- **TensorFlow Datasets**: Data loading and management

### Mobile Dependencies (planned)
- **Flutter**: Cross-platform mobile development framework
- **TensorFlow Lite Flutter Plugin**: For running .tflite models on mobile devices

## Deployment Strategy

### Current State - Ready for Training
- **Training Pipeline**: Complete weather_model_trainer.py with EfficientNetB0 architecture
- **Data Structure**: weather_images/ with train_dir and test_dir folders for Cloudy, Rain, Sunrise, Shine
- **Flutter Integration**: Complete MLService with TensorFlow Lite support and intelligent image analysis
- **Model Conversion**: Automated conversion to mobile-optimized .tflite format

### Training Process
1. **Add Images**: Place weather images in weather_images/train_dir/[Cloudy|Rain|Sunrise|Shine]/
2. **Run Training**: Execute `python weather_model_trainer.py` 
3. **Model Output**: Generates optimized weather_model.tflite for Flutter app
4. **Integration**: Model automatically loads in Flutter MLService with real-time classification

### Mobile Deployment Features
- **Real-time Classification**: Camera and gallery image processing
- **Intelligent Analysis**: Image preprocessing with brightness and color analysis
- **AI Recommendations**: Context-aware suggestions based on detected weather
- **Professional UI**: Confidence scores, gradient design, mobile-optimized interface

### Key Architectural Decisions

1. **EfficientNetB0 Choice**: Balances accuracy and efficiency for mobile deployment
2. **Transfer Learning**: Frozen pre-trained weights reduce training time and improve performance
3. **TensorFlow Lite**: Enables on-device inference without internet dependency
4. **Standardized Input Size**: 224x224 pixels optimized for EfficientNet architecture
5. **4-Class Classification**: Focused on common weather conditions for practical use cases

The architecture prioritizes mobile efficiency while maintaining classification accuracy through proven transfer learning techniques.