"""
SageMaker Inference Handler for CIFAR-10 CNN Model
===================================================
This file handles model loading, input parsing, prediction, and output formatting
for the CIFAR-10 CNN classification model deployed on AWS SageMaker.

The model expects 32x32x3 RGB images normalized to [0, 1] range and outputs
predictions across 10 classes.
"""

import json
import numpy as np
import os


# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def model_fn(model_dir):
    """
    Load the trained Keras CNN model from disk.
    Called once when the endpoint starts.

    Args:
        model_dir: Path to the model directory containing the SavedModel

    Returns:
        Loaded TensorFlow/Keras model
    """
    import tensorflow as tf

    model_path = os.path.join(model_dir, "cifar10_cnn_model")

    # Try loading as SavedModel directory first
    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from SavedModel directory: {model_path}")
    else:
        # Fallback: try .keras or .h5 format
        for ext in ['.keras', '.h5']:
            alt_path = os.path.join(model_dir, f"cifar10_cnn_model{ext}")
            if os.path.exists(alt_path):
                model = tf.keras.models.load_model(alt_path)
                print(f"Model loaded from: {alt_path}")
                break
        else:
            raise FileNotFoundError(
                f"No model found in {model_dir}. "
                f"Expected 'cifar10_cnn_model/' directory, "
                f"'cifar10_cnn_model.keras', or 'cifar10_cnn_model.h5'"
            )

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Parse incoming request data.

    Accepts JSON with either:
      - "instances": list of images as 3D arrays (batch)
      - "image": single image as 3D array
      - "images": list of images as 3D arrays (batch)

    Images should be pixel values in [0, 255] or already normalized [0, 1].

    Args:
        request_body: Raw request body
        content_type: Content type of the request

    Returns:
        NumPy array of shape (batch_size, 32, 32, 3) normalized to [0, 1]
    """
    if content_type == 'application/json':
        data = json.loads(request_body)

        if 'instances' in data:
            images = np.array(data['instances'], dtype=np.float32)
        elif 'images' in data:
            images = np.array(data['images'], dtype=np.float32)
        elif 'image' in data:
            images = np.array(data['image'], dtype=np.float32)
            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)
        else:
            raise ValueError(
                "JSON must contain 'instances', 'images', or 'image' key"
            )

        # Normalize to [0, 1] if values are in [0, 255]
        if images.max() > 1.0:
            images = images / 255.0

        # Validate shape
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(
                f"Expected images of shape (batch, 32, 32, 3), "
                f"got {images.shape}"
            )

        return images
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded CNN model.

    Args:
        input_data: NumPy array of shape (batch_size, 32, 32, 3)
        model: Loaded Keras model

    Returns:
        List of prediction dictionaries with class probabilities,
        predicted class, and confidence score
    """
    # Run inference
    predictions = model.predict(input_data, verbose=0)

    results = []
    for pred in predictions:
        predicted_class_idx = int(np.argmax(pred))
        confidence = float(pred[predicted_class_idx])

        # Build top-3 predictions
        top3_indices = np.argsort(pred)[::-1][:3]
        top3 = [
            {
                'class': CLASS_NAMES[idx],
                'class_index': int(idx),
                'probability': float(pred[idx])
            }
            for idx in top3_indices
        ]

        results.append({
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'predicted_class_index': predicted_class_idx,
            'confidence': confidence,
            'top_3_predictions': top3,
            'all_probabilities': {
                CLASS_NAMES[i]: round(float(pred[i]), 6)
                for i in range(len(CLASS_NAMES))
            }
        })

    return results


def output_fn(prediction, accept='application/json'):
    """
    Format the prediction response.

    Args:
        prediction: List of prediction dictionaries
        accept: Accepted content type

    Returns:
        JSON string of the predictions
    """
    if accept == 'application/json':
        return json.dumps(prediction, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
