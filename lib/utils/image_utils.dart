import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImageUtils {
  static const int MODEL_INPUT_SIZE = 224;

  /// Preprocesses an image for the weather classification model
  static Future<Float32List?> preprocessImageForModel(String imagePath) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        print('Image file does not exist: $imagePath');
        return null;
      }

      // Read image bytes
      final bytes = await file.readAsBytes();
      
      // Decode image
      img.Image? image = img.decodeImage(bytes);
      if (image == null) {
        print('Failed to decode image');
        return null;
      }

      // Resize to model input size
      image = img.copyResize(
        image,
        width: MODEL_INPUT_SIZE,
        height: MODEL_INPUT_SIZE,
        interpolation: img.Interpolation.linear,
      );

      // Convert to Float32List with normalization
      final Float32List input = Float32List(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
      int pixelIndex = 0;

      for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
        for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
          final pixel = image.getPixel(x, y);
          
          // Normalize pixel values to [0, 1] range
          input[pixelIndex++] = img.getRed(pixel) / 255.0;
          input[pixelIndex++] = img.getGreen(pixel) / 255.0;
          input[pixelIndex++] = img.getBlue(pixel) / 255.0;
        }
      }

      return input;
    } catch (e) {
      print('Error preprocessing image: $e');
      return null;
    }
  }

  /// Validates if an image file is valid and accessible
  static Future<bool> validateImageFile(String imagePath) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        return false;
      }

      // Check file size (should be reasonable, not too large or too small)
      final fileSize = await file.length();
      if (fileSize < 1024 || fileSize > 50 * 1024 * 1024) { // 1KB to 50MB
        return false;
      }

      // Try to decode the image to ensure it's valid
      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);
      
      return image != null;
    } catch (e) {
      print('Error validating image file: $e');
      return false;
    }
  }

  /// Compresses an image to reduce file size while maintaining quality
  static Future<String?> compressImage(String imagePath, {int quality = 85}) async {
    try {
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      
      img.Image? image = img.decodeImage(bytes);
      if (image == null) return null;

      // Resize if image is too large
      if (image.width > 1024 || image.height > 1024) {
        final ratio = image.width / image.height;
        int newWidth, newHeight;
        
        if (ratio > 1) {
          newWidth = 1024;
          newHeight = (1024 / ratio).round();
        } else {
          newHeight = 1024;
          newWidth = (1024 * ratio).round();
        }
        
        image = img.copyResize(image, width: newWidth, height: newHeight);
      }

      // Encode with compression
      final compressedBytes = img.encodeJpg(image, quality: quality);
      
      // Save compressed image
      final compressedPath = imagePath.replaceAll('.', '_compressed.');
      final compressedFile = File(compressedPath);
      await compressedFile.writeAsBytes(compressedBytes);
      
      return compressedPath;
    } catch (e) {
      print('Error compressing image: $e');
      return null;
    }
  }

  /// Gets image dimensions without loading the full image into memory
  static Future<Map<String, int>?> getImageDimensions(String imagePath) async {
    try {
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      
      final image = img.decodeImage(bytes);
      if (image == null) return null;
      
      return {
        'width': image.width,
        'height': image.height,
      };
    } catch (e) {
      print('Error getting image dimensions: $e');
      return null;
    }
  }

  /// Creates a thumbnail from an image
  static Future<String?> createThumbnail(String imagePath, {int size = 150}) async {
    try {
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      
      img.Image? image = img.decodeImage(bytes);
      if (image == null) return null;

      // Create square thumbnail
      final thumbnail = img.copyResizeCropSquare(image, size: size);
      
      // Save thumbnail
      final thumbnailPath = imagePath.replaceAll('.', '_thumb.');
      final thumbnailBytes = img.encodeJpg(thumbnail, quality: 80);
      
      final thumbnailFile = File(thumbnailPath);
      await thumbnailFile.writeAsBytes(thumbnailBytes);
      
      return thumbnailPath;
    } catch (e) {
      print('Error creating thumbnail: $e');
      return null;
    }
  }
}
