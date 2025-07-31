"""
Weather AI Model Training Script
Trains a deep learning model on real weather images for classification.
Based on the provided training code structure with EfficientNetB0.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 25
EPOCHS_FINE_TUNE = 10
NUM_CLASSES = 4
WEATHER_CLASSES = ['Cloudy', 'Rain', 'Sunrise', 'Shine']

# Data paths
TRAIN_DIR = 'weather_images/train_dir'
TEST_DIR = 'weather_images/test_dir'
MODEL_SAVE_PATH = 'weather_model.h5'
TFLITE_SAVE_PATH = 'assets/models/weather_model.tflite'

def check_data_structure():
    """Verify the data directory structure and count images"""
    print("Checking data structure...")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        return False
    
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        return False
    
    train_counts = {}
    test_counts = {}
    
    for weather_class in WEATHER_CLASSES:
        train_path = os.path.join(TRAIN_DIR, weather_class)
        test_path = os.path.join(TEST_DIR, weather_class)
        
        if os.path.exists(train_path):
            train_counts[weather_class] = len([f for f in os.listdir(train_path) 
                                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            train_counts[weather_class] = 0
            print(f"⚠️  Training folder missing: {train_path}")
        
        if os.path.exists(test_path):
            test_counts[weather_class] = len([f for f in os.listdir(test_path) 
                                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            test_counts[weather_class] = 0
            print(f"⚠️  Test folder missing: {test_path}")
    
    print("\n📊 Dataset Summary:")
    print("=" * 50)
    total_train = 0
    total_test = 0
    
    for weather_class in WEATHER_CLASSES:
        train_count = train_counts[weather_class]
        test_count = test_counts[weather_class]
        total_train += train_count
        total_test += test_count
        
        print(f"{weather_class:12}: {train_count:4} train | {test_count:4} test")
    
    print("-" * 50)
    print(f"{'Total':12}: {total_train:4} train | {total_test:4} test")
    
    if total_train == 0:
        print("\n❌ No training images found! Please add images to the weather_images/train_dir folders.")
        return False
    
    if total_test == 0:
        print("\n⚠️  No test images found. Using train/validation split instead.")
    
    return True

def load_and_preprocess_data():
    """Load and preprocess the weather image data"""
    print("\n🔄 Loading and preprocessing data...")
    
    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # Use 20% for validation if no test set
    )
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    
    # Load test data if available
    test_generator = None
    if os.path.exists(TEST_DIR) and any(os.listdir(os.path.join(TEST_DIR, cls)) for cls in WEATHER_CLASSES):
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    if test_generator:
        print(f"✅ Test samples: {test_generator.samples}")
    
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator, test_generator

def visualize_data_samples(train_generator):
    """Display sample images from each class"""
    print("\n📸 Visualizing sample data...")
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Sample Weather Images', fontsize=16)
    
    class_names = list(train_generator.class_indices.keys())
    
    # Get samples for each class
    for i, class_name in enumerate(class_names):
        # Find the first image of this class
        found = False
        for batch_x, batch_y in train_generator:
            for j in range(len(batch_y)):
                if np.argmax(batch_y[j]) == i and not found:
                    # Display original
                    axes[0, i].imshow(batch_x[j])
                    axes[0, i].set_title(f'{class_name} (Original)')
                    axes[0, i].axis('off')
                    
                    # Display augmented version
                    if j + 1 < len(batch_x):
                        axes[1, i].imshow(batch_x[j + 1])
                        axes[1, i].set_title(f'{class_name} (Augmented)')
                        axes[1, i].axis('off')
                    
                    found = True
                    break
            if found:
                break
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Sample visualization saved as 'data_samples.png'")
    plt.close()

def build_weather_model():
    """Build the weather classification model using EfficientNetB0"""
    print("\n🏗️  Building model architecture...")
    
    # Input layer
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Base model - EfficientNetB0 pre-trained on ImageNet
    base_model = EfficientNetB0(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet"
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    
    # Create the model
    model = keras.Model(inputs, outputs, name="WeatherClassifier")
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("✅ Model architecture built successfully")
    print(f"✅ Total parameters: {model.count_params():,}")
    print(f"✅ Trainable parameters: {sum([np.prod(v.get_shape()) for v in model.trainable_weights]):,}")
    
    return model

def train_model_phase1(model, train_generator, validation_generator):
    """Phase 1: Train with frozen base model"""
    print(f"\n🎯 Phase 1: Training with frozen base model ({EPOCHS_INITIAL} epochs)")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model_phase1.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_INITIAL,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history1

def fine_tune_model(model, train_generator, validation_generator):
    """Phase 2: Fine-tune with unfrozen layers"""
    print(f"\n🎯 Phase 2: Fine-tuning model ({EPOCHS_FINE_TUNE} epochs)")
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[1]  # EfficientNetB0 layer
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 20
    
    # Freeze all the layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"✅ Unfrozen layers: {sum([1 for layer in model.layers if layer.trainable])}")
    print(f"✅ Trainable parameters: {sum([np.prod(v.get_shape()) for v in model.trainable_weights]):,}")
    
    # Callbacks for fine-tuning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Continue training
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history2

def plot_training_history(history1, history2=None):
    """Plot training and validation metrics"""
    print("\n📊 Plotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Combine histories if fine-tuning was performed
    if history2:
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        epochs = range(1, len(acc) + 1)
        phase1_end = len(history1.history['accuracy'])
    else:
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']
        epochs = range(1, len(acc) + 1)
        phase1_end = None
    
    # Accuracy plot
    axes[0, 0].plot(epochs, acc, 'b-', label='Training Accuracy')
    axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    if phase1_end:
        axes[0, 0].axvline(x=phase1_end, color='g', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(epochs, loss, 'b-', label='Training Loss')
    axes[0, 1].plot(epochs, val_loss, 'r-', label='Validation Loss')
    if phase1_end:
        axes[0, 1].axvline(x=phase1_end, color='g', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final metrics
    final_acc = acc[-1]
    final_val_acc = val_acc[-1]
    final_loss = loss[-1]
    final_val_loss = val_loss[-1]
    
    axes[1, 0].text(0.1, 0.8, f'Final Training Accuracy: {final_acc:.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.7, f'Final Validation Accuracy: {final_val_acc:.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.6, f'Final Training Loss: {final_loss:.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.5, f'Final Validation Loss: {final_val_loss:.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Final Metrics')
    axes[1, 0].axis('off')
    
    # Training summary
    axes[1, 1].text(0.1, 0.8, f'Total Epochs: {len(epochs)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Best Validation Accuracy: {max(val_acc):.4f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Image Size: {IMG_SIZE}x{IMG_SIZE}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Batch Size: {BATCH_SIZE}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Classes: {NUM_CLASSES}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Training Configuration')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Training history saved as 'training_history.png'")
    plt.close()

def evaluate_model(model, test_generator):
    """Evaluate the trained model"""
    print("\n📋 Evaluating model performance...")
    
    if test_generator is None:
        print("⚠️  No test data available for evaluation")
        return
    
    # Make predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Calculate accuracy
    test_accuracy = np.mean(predicted_classes == true_classes)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print("=" * 50)
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()

def convert_to_tflite(model_path):
    """Convert the trained model to TensorFlow Lite format"""
    print(f"\n🔄 Converting model to TensorFlow Lite...")
    
    # Load the trained model
    model = keras.models.load_model(model_path)
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        
        # Save the model
        os.makedirs(os.path.dirname(TFLITE_SAVE_PATH), exist_ok=True)
        with open(TFLITE_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to: {TFLITE_SAVE_PATH}")
        print(f"✅ Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        # Test the converted model
        test_tflite_model(TFLITE_SAVE_PATH)
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

def test_tflite_model(tflite_path):
    """Test the converted TFLite model"""
    print(f"\n🧪 Testing TFLite model...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Input shape: {input_details[0]['shape']}")
        print(f"✅ Output shape: {output_details[0]['shape']}")
        
        # Test with random input
        input_shape = input_details[0]['shape']
        input_data = np.random.random_sample(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ Test inference successful")
        print(f"✅ Sample output: {output_data[0]}")
        
    except Exception as e:
        print(f"❌ Error testing TFLite model: {e}")

def main():
    """Main training pipeline"""
    print("🌤️  Weather AI Model Training Pipeline")
    print("=" * 60)
    
    # Check data structure
    if not check_data_structure():
        print("\n❌ Please ensure your data is properly organized:")
        print("weather_images/")
        print("├── train_dir/")
        print("│   ├── Cloudy/")
        print("│   ├── Rain/")
        print("│   ├── Sunrise/")
        print("│   └── Shine/")
        print("└── test_dir/")
        print("    ├── Cloudy/")
        print("    ├── Rain/")
        print("    ├── Sunrise/")
        print("    └── Shine/")
        return
    
    # Load and preprocess data
    train_gen, val_gen, test_gen = load_and_preprocess_data()
    
    # Visualize samples
    visualize_data_samples(train_gen)
    
    # Build model
    model = build_weather_model()
    
    # Phase 1: Train with frozen base
    history1 = train_model_phase1(model, train_gen, val_gen)
    
    # Phase 2: Fine-tune
    history2 = fine_tune_model(model, train_gen, val_gen)
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Evaluate model
    if test_gen:
        evaluate_model(model, test_gen)
    
    # Convert to TFLite
    convert_to_tflite(MODEL_SAVE_PATH)
    
    print("\n🎉 Training pipeline completed!")
    print(f"✅ Model saved as: {MODEL_SAVE_PATH}")
    print(f"✅ TFLite model saved as: {TFLITE_SAVE_PATH}")
    print("✅ Ready for integration with Flutter app!")

if __name__ == "__main__":
    main()