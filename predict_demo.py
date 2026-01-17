"""
Endangered Wildlife Image Classification - Interactive Prediction Demo
SAIA 2133: Computer Vision - Universiti Teknologi Malaysia (UTM)

This script provides a standalone prediction tool for classifying wildlife images
using the trained models from the main notebook.

Usage:
    python predict_demo.py --image path/to/image.jpg --model custom_cnn
    python predict_demo.py --image path/to/image.jpg --model mobilenetv2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
IMG_SIZE = (224, 224)
MODELS_DIR = Path('models')

# Class names for Malaysia-specific endangered species (4 classes)
CLASS_NAMES = ['Elephant', 'Orangutan', 'Panthers', 'Rhino']


def load_model(model_name):
    """Load a trained model"""
    # Map model names to actual filenames saved by the notebook
    if model_name == "mobilenetv2":
        filename = "mobilenetv2_transfer_final.h5"
    elif model_name == "custom_cnn":
        filename = "custom_cnn_final.h5"
    else:
        filename = f"{model_name}_final.h5"
    
    model_path = MODELS_DIR / filename
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the models first by running the Jupyter notebook:\n"
            f"  notebooks/wildlife_classification.ipynb"
        )
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img, img_array


def predict_and_visualize(image_path, model, class_names, model_name="Model"):
    """
    Predict wildlife class from an image and display results
    
    Args:
        image_path: Path to image file
        model: Trained Keras model
        class_names: List of class names
        model_name: Name of model for display
    """
    # Preprocess image
    img, img_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Display image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f'Input Image\nPredicted: {predicted_class}\nConfidence: {confidence:.2f}%',
                     fontsize=12, fontweight='bold', 
                     color='green' if confidence > 80 else 'orange')
    
    # Display prediction probabilities
    sorted_indices = np.argsort(predictions[0])[::-1]
    top_classes = [class_names[i] for i in sorted_indices]
    top_probs = [predictions[0][i] * 100 for i in sorted_indices]
    
    colors = ['green' if i == predicted_class_idx else 'lightblue' 
              for i in sorted_indices]
    axes[1].barh(top_classes, top_probs, color=colors, edgecolor='black')
    axes[1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1].set_title(f'{model_name} Prediction Probabilities', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 100])
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(top_probs):
        axes[1].text(v + 1, i, f'{v:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save output
    output_path = Path('results') / f'prediction_{Path(image_path).stem}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    # Print results
    print("\n" + "="*60)
    print(f"{model_name} Prediction Results")
    print("="*60)
    print(f"Image: {Path(image_path).name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nTop 3 Predictions:")
    for i in range(min(3, len(class_names))):
        print(f"  {i+1}. {top_classes[i]}: {top_probs[i]:.2f}%")
    print("="*60)
    
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description='Predict wildlife class from image using trained models'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['custom_cnn', 'mobilenetv2'],
        default='mobilenetv2',
        help='Model to use for prediction (default: mobilenetv2)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        help='List of class names (space-separated)'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return
    
    # Load class names
    if args.classes:
        class_names = args.classes
    else:
        class_names = CLASS_NAMES
        print(f"Using default class names: {class_names}")
        print("   To specify custom classes, use: --classes Elephant Orangutan Panthers Rhino")
    
    # Load model
    try:
        model = load_model(args.model)
        model_display_name = "Custom CNN" if args.model == "custom_cnn" else "MobileNetV2 Transfer"
        
        print(f"\nLoaded {model_display_name} model")
        print(f"   Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return
    
    # Make prediction
    try:
        predict_and_visualize(image_path, model, class_names, model_display_name)
    except Exception as e:
        print(f"ERROR: Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
