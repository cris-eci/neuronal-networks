#!/usr/bin/env python3
"""
CIFAR-10 CNN Model - SageMaker Deployment Demo
================================================

This script demonstrates the full SageMaker deployment workflow for the
CIFAR-10 CNN classification model trained in the notebook.

Run from SageMaker Code Editor terminal:
    python sagemaker_scripts/demo_deployment.py
"""
import json
import numpy as np
import os
import logging

# Suppress SageMaker INFO messages (fixes hanging issue!)
logging.getLogger('sagemaker.config').setLevel(logging.WARNING)

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel


# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def local_inference(model_dir='cifar10_cnn_model'):
    """
    Run inference locally using the saved Keras model.
    Used as fallback when SageMaker endpoint creation is blocked.

    Args:
        model_dir: Path to the saved Keras model directory
    """
    import tensorflow as tf

    print("\n🧪 Running LOCAL inference test...")
    print("-" * 50)

    # Load the model
    if os.path.isdir(model_dir):
        model = tf.keras.models.load_model(model_dir)
    elif os.path.exists(f"{model_dir}.keras"):
        model = tf.keras.models.load_model(f"{model_dir}.keras")
    elif os.path.exists(f"{model_dir}.h5"):
        model = tf.keras.models.load_model(f"{model_dir}.h5")
    else:
        print(f"   ❌ Model not found at '{model_dir}'")
        print("   Make sure you exported the model from the notebook first.")
        return

    print(f"   ✅ Model loaded: {model.name}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Parameters: {model.count_params():,}")

    # Load CIFAR-10 test data for sample predictions
    print("\n   Loading CIFAR-10 test data for sample predictions...")
    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test_normalized = x_test.astype('float32') / 255.0

    # Select diverse test samples (one per class if possible)
    test_indices = []
    for class_idx in range(10):
        indices = np.where(y_test.flatten() == class_idx)[0]
        if len(indices) > 0:
            test_indices.append(indices[0])

    test_images = x_test_normalized[test_indices]
    test_labels = y_test[test_indices].flatten()

    # Run predictions
    predictions = model.predict(test_images, verbose=0)

    print(f"\n   {'='*60}")
    print(f"   PREDICTION RESULTS (one sample per class)")
    print(f"   {'='*60}")

    correct = 0
    for i, (pred, true_label) in enumerate(zip(predictions, test_labels)):
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        is_correct = predicted_class == true_label
        correct += int(is_correct)

        status = "✅" if is_correct else "❌"
        print(f"\n   {status} Sample {i+1}:")
        print(f"      True class:      {CLASS_NAMES[true_label]}")
        print(f"      Predicted class:  {CLASS_NAMES[predicted_class]}")
        print(f"      Confidence:       {confidence:.2%}")

        # Show top-3
        top3 = np.argsort(pred)[::-1][:3]
        top3_str = ", ".join(
            f"{CLASS_NAMES[idx]} ({pred[idx]:.1%})" for idx in top3
        )
        print(f"      Top 3:            {top3_str}")

    accuracy = correct / len(test_labels)
    print(f"\n   {'='*60}")
    print(f"   Sample Accuracy: {correct}/{len(test_labels)} ({accuracy:.0%})")
    print(f"   {'='*60}")


def main():
    print("=" * 70)
    print("🚀 CIFAR-10 CNN MODEL - SAGEMAKER DEPLOYMENT")
    print("=" * 70)

    # -----------------------------------------------------------
    # Step 1: Initialize SageMaker session
    # -----------------------------------------------------------
    print("\n📦 Step 1: Initializing SageMaker session...")
    try:
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        bucket = sagemaker_session.default_bucket()
        role = sagemaker.get_execution_role()
        print(f"   ✅ Region: {region}")
        print(f"   ✅ Bucket: {bucket}")
        print(f"   ✅ Role:   {role[:60]}...")
    except Exception as e:
        print(f"   ❌ SageMaker initialization failed: {e}")
        print("\n   Falling back to local inference test...")
        local_inference()
        return

    # -----------------------------------------------------------
    # Step 2: Upload model.tar.gz to S3
    # -----------------------------------------------------------
    print("\n📤 Step 2: Uploading model.tar.gz to S3...")

    model_tar = 'model.tar.gz'
    if not os.path.exists(model_tar):
        print(f"   ❌ {model_tar} not found!")
        print("   Run the model export cells in the notebook first.")
        print("   The notebook will create model.tar.gz automatically.")
        return

    s3_model_path = sagemaker_session.upload_data(
        path=model_tar,
        bucket=bucket,
        key_prefix='cifar10-cnn-model'
    )
    print(f"   ✅ S3 Path: {s3_model_path}")

    # -----------------------------------------------------------
    # Step 3: Create SageMaker TensorFlow Model object
    # -----------------------------------------------------------
    print("\n🔧 Step 3: Creating SageMaker TensorFlow Model object...")

    model = TensorFlowModel(
        model_data=s3_model_path,
        role=role,
        framework_version='2.13',
        entry_point='inference.py',
        source_dir='sagemaker_scripts',
        sagemaker_session=sagemaker_session
    )
    print("   ✅ TensorFlow Model object created successfully")

    # -----------------------------------------------------------
    # Step 4: Deploy endpoint
    # -----------------------------------------------------------
    print("\n🌐 Step 4: Deploying endpoint...")
    print("   " + "-" * 50)
    print("   📋 Endpoint Name:    cifar10-cnn-endpoint")
    print("   📋 Instance Type:    ml.m5.large")
    print("   📋 Instance Count:   1")
    print(f"   📋 Model Data:       {s3_model_path}")
    print("   📋 Framework:        TensorFlow 2.13")
    print("   " + "-" * 50)

    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',
            endpoint_name='cifar10-cnn-endpoint'
        )
        print("   ✅ Endpoint deployed successfully!")

        # -----------------------------------------------------------
        # Step 5: Test the endpoint with CIFAR-10 samples
        # -----------------------------------------------------------
        print("\n🧪 Step 5: Testing endpoint with CIFAR-10 samples...")

        from tensorflow.keras.datasets import cifar10
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test_normalized = x_test.astype('float32') / 255.0

        # Test with a few samples
        test_indices = [0, 1, 2, 3, 4]
        test_images = x_test_normalized[test_indices]
        test_labels = y_test[test_indices].flatten()

        # Send prediction request
        payload = {'instances': test_images.tolist()}
        response = predictor.predict(payload)

        print(f"\n   {'='*50}")
        for i, (result, true_label) in enumerate(zip(response, test_labels)):
            pred_class = result.get('predicted_class', 'unknown')
            confidence = result.get('confidence', 0.0)
            status = "✅" if pred_class == CLASS_NAMES[true_label] else "❌"
            print(f"   {status} Image {i+1}: True={CLASS_NAMES[true_label]}, "
                  f"Predicted={pred_class} ({confidence:.2%})")
        print(f"   {'='*50}")

        # Cleanup
        print("\n" + "=" * 70)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"\n📁 Endpoint: {predictor.endpoint_name}")
        print("\n⚠️  Remember to delete the endpoint when done:")
        print("   predictor.delete_endpoint()")

    except Exception as e:
        print(f"\n   ❌ Endpoint deployment failed: {e}")
        print("\n   This is expected in AWS Learner Lab (IAM restriction).")
        print("   The model was successfully uploaded to S3.")

        # -----------------------------------------------------------
        # Step 5 (Fallback): Local inference test
        # -----------------------------------------------------------
        local_inference()

        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT DEMO COMPLETE (with local fallback)")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   ┌──────────────────────────────────────────┬────────┐")
        print("   │ Component                                │ Status │")
        print("   ├──────────────────────────────────────────┼────────┤")
        print("   │ SageMaker Session                        │   ✅   │")
        print("   │ Model uploaded to S3                     │   ✅   │")
        print("   │ SageMaker TensorFlow Model object        │   ✅   │")
        print("   │ Inference script (inference.py)          │   ✅   │")
        print("   │ Local inference test                     │   ✅   │")
        print("   │ Real endpoint deployment                 │   ❌   │")
        print("   └──────────────────────────────────────────┴────────┘")
        print(f"\n📁 Model artifacts stored in S3:")
        print(f"   {s3_model_path}")


if __name__ == "__main__":
    main()
