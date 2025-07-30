"""
Model converter script to convert TensorFlow model to TensorFlow Lite
for Flutter mobile deployment.

This script should be run separately to generate the .tflite model file
that will be used in the Flutter app.
"""

import tensorflow as tf
import numpy as np
from keras import layers
from keras.applications import EfficientNetB0
import os

# Model parameters (should match the training script)
IMG_SIZE = 224
NUM_CLASSES = 4  # Cloudy, Rain, Sunrise, Shine

def build_model(num_classes):
    """Build the same model architecture used in training"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    return model

def convert_to_tflite(saved_model_path, output_path):
    """Convert saved TensorFlow model to TensorFlow Lite format"""
    
    # Load the saved model
    if os.path.isdir(saved_model_path):
        # If it's a SavedModel directory
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        # If it's a .h5 or .keras file
        model = tf.keras.models.load_model(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for quantization (optional but recommended)
    def representative_data_gen():
        for _ in range(100):
            # Generate random data that matches your input shape
            yield [np.random.random((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model successfully converted and saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        # Test the converted model
        test_tflite_model(output_path)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        
        # Try without quantization as fallback
        print("Trying conversion without quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model) if 'model' in locals() else tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted without quantization and saved to: {output_path}")
        test_tflite_model(output_path)

def test_tflite_model(tflite_path):
    """Test the converted TFLite model to ensure it works correctly"""
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Test with random input
        input_shape = input_details[0]['shape']
        input_data = np.random.random_sample(input_shape).astype(input_details[0]['dtype'])
        
        if input_details[0]['dtype'] == np.uint8:
            input_data = (input_data * 255).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Test inference successful. Output shape: {output_data.shape}")
        print(f"Sample output: {output_data[0] if len(output_data.shape) > 1 else output_data}")
        
    except Exception as e:
        print(f"Error testing TFLite model: {e}")

def create_sample_model():
    """Create a sample model for testing if no trained model is available"""
    print("Creating sample model for testing...")
    
    model = build_model(NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy data for a quick training (just for testing)
    dummy_x = np.random.random((32, IMG_SIZE, IMG_SIZE, 3))
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 32), NUM_CLASSES)
    
    # Train for just 1 epoch to have some weights
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    # Save the model
    model.save('sample_weather_model.h5')
    print("Sample model created and saved as 'sample_weather_model.h5'")
    
    return 'sample_weather_model.h5'

def main():
    """Main conversion function"""
    print("Weather AI Model Converter")
    print("=" * 40)
    
    # Check for existing trained model
    possible_model_paths = [
        'weather_model.h5',
        'weather_model.keras',
        'weather_model_saved',
        'model.h5',
        'trained_model.h5'
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("No trained model found. Creating a sample model for testing...")
        model_path = create_sample_model()
    else:
        print(f"Found trained model: {model_path}")
    
    # Output path for TFLite model
    output_path = 'assets/models/weather_model.tflite'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert model
    convert_to_tflite(model_path, output_path)
    
    print("\nConversion complete!")
    print(f"TFLite model saved to: {output_path}")
    print("\nNext steps:")
    print("1. Copy the generated .tflite file to your Flutter app's assets/models/ folder")
    print("2. Make sure the model file is listed in your pubspec.yaml under assets")
    print("3. The Flutter app will load this model for weather classification")

if __name__ == "__main__":
    main()
