"""
Bonus Visualization Generator for CNN Assignment
=================================================

This script generates advanced visualizations for the README bonus section:
- Learned convolutional filters
- Feature map activations
- Prediction confidence distribution
- t-SNE visualization of learned features

Run this after training your CNN model in the notebook.

Usage:
------
In your Jupyter notebook, after training:
    
    %run generate_bonus_visualizations.py

Or import and run specific functions:
    
    from generate_bonus_visualizations import generate_all_visualizations
    generate_all_visualizations(cnn_model, x_test_normalized, y_test, class_names)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import os


def create_images_directory():
    """Create images directory if it doesn't exist."""
    os.makedirs('images', exist_ok=True)
    print("✓ Images directory created/verified")


def visualize_conv1_filters(model, save_path='images/conv1_filters.png'):
    """
    Visualize learned filters from the first convolutional layer.
    
    Parameters:
    -----------
    model : keras.Model
        Trained CNN model
    save_path : str
        Path to save the visualization
    """
    print("\n📊 Generating Conv1 filter visualization...")
    
    # Get weights from first convolutional layer
    conv1_layer = None
    for layer in model.layers:
        if 'conv' in layer.name.lower() and len(layer.get_weights()) > 0:
            conv1_layer = layer
            break
    
    if conv1_layer is None:
        print("❌ Could not find convolutional layer")
        return
    
    weights = conv1_layer.get_weights()[0]  # Shape: (kernel_h, kernel_w, channels, num_filters)
    num_filters = weights.shape[-1]
    
    # Calculate grid dimensions
    grid_cols = 8
    grid_rows = (num_filters + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, grid_rows * 2))
    axes = axes.flatten() if num_filters > 1 else [axes]
    
    for i in range(num_filters):
        # Extract and normalize filter
        filt = weights[:, :, :, i]
        
        # Normalize to [0, 1] for visualization
        filt_min, filt_max = filt.min(), filt.max()
        if filt_max > filt_min:
            filt = (filt - filt_min) / (filt_max - filt_min)
        
        # Display filter
        axes[i].imshow(filt)
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}', fontsize=8, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{conv1_layer.name.upper()}: Learned Filters ({weights.shape[0]}×{weights.shape[1]}×{weights.shape[2]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.show()


def visualize_feature_maps(model, sample_images, save_path='images/feature_maps.png'):
    """
    Visualize feature map activations across convolutional layers.
    
    Parameters:
    -----------
    model : keras.Model
        Trained CNN model
    sample_images : numpy.ndarray
        Preprocessed images to visualize (shape: (N, H, W, C))
    save_path : str
        Path to save the visualization
    """
    print("\n📊 Generating feature map visualization...")
    
    # Ensure model has been called at least once
    # This is necessary to define the input/output structure
    _ = model.predict(sample_images[0:1], verbose=0)
    
    # Find all convolutional layers
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
    
    if not conv_layers:
        print("❌ No convolutional layers found")
        return
    
    # Create model to output intermediate activations
    layer_outputs = [layer.output for layer in conv_layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations for first sample image
    sample_input = sample_images[0:1]  # Shape: (1, H, W, C)
    activations = activation_model.predict(sample_input, verbose=0)
    
    # Visualize first 8 feature maps from each layer
    num_layers = len(conv_layers)
    num_features = 8
    
    fig, axes = plt.subplots(num_layers, num_features, figsize=(16, num_layers * 2))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, (activation, layer) in enumerate(zip(activations, conv_layers)):
        for feat_idx in range(num_features):
            ax = axes[layer_idx, feat_idx]
            
            # Handle case where layer has fewer than 8 filters
            if feat_idx < activation.shape[-1]:
                feature_map = activation[0, :, :, feat_idx]
                ax.imshow(feature_map, cmap='viridis')
            else:
                ax.imshow(np.zeros((activation.shape[1], activation.shape[2])), cmap='gray')
            
            ax.axis('off')
            
            # Add titles
            if feat_idx == 0:
                ax.text(-0.1, 0.5, f'{layer.name}\n({activation.shape[1]}×{activation.shape[2]}×{activation.shape[3]})',
                       transform=ax.transAxes, fontsize=9, fontweight='bold',
                       verticalalignment='center', rotation=0)
            
            if layer_idx == 0:
                ax.set_title(f'Feature {feat_idx + 1}', fontsize=9)
    
    plt.suptitle('Feature Map Activations Across Convolutional Layers', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.show()


def visualize_confidence_distribution(model, x_test, y_test, save_path='images/confidence_distribution.png'):
    """
    Visualize prediction confidence distribution for correct vs incorrect predictions.
    
    Parameters:
    -----------
    model : keras.Model
        Trained CNN model
    x_test : numpy.ndarray
        Test images
    y_test : numpy.ndarray
        Test labels
    save_path : str
        Path to save the visualization
    """
    print("\n📊 Generating confidence distribution analysis...")
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    max_probs = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test.flatten() if len(y_test.shape) > 1 else y_test
    
    # Separate correct and incorrect predictions
    correct_mask = (predicted_classes == true_classes)
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = np.linspace(0, 1, 51)
    ax.hist(correct_confidences, bins=bins, alpha=0.7, label='Correct Predictions', 
            color='#2E7D32', edgecolor='black', linewidth=0.5)
    ax.hist(incorrect_confidences, bins=bins, alpha=0.7, label='Incorrect Predictions', 
            color='#C62828', edgecolor='black', linewidth=0.5)
    
    # Add statistics
    ax.axvline(np.mean(correct_confidences), color='#2E7D32', linestyle='--', 
              linewidth=2, label=f'Mean (Correct): {np.mean(correct_confidences):.3f}')
    ax.axvline(np.mean(incorrect_confidences), color='#C62828', linestyle='--', 
              linewidth=2, label=f'Mean (Incorrect): {np.mean(incorrect_confidences):.3f}')
    
    ax.set_xlabel('Maximum Softmax Probability (Confidence)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution: Correct vs Incorrect', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    correct_count = np.sum(correct_mask)
    incorrect_count = np.sum(~correct_mask)
    stats_text = f'Correct: {correct_count:,} ({correct_count/len(y_test)*100:.1f}%)\n'
    stats_text += f'Incorrect: {incorrect_count:,} ({incorrect_count/len(y_test)*100:.1f}%)'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 Confidence Statistics:")
    print(f"   Correct predictions - Mean: {np.mean(correct_confidences):.3f}, Median: {np.median(correct_confidences):.3f}")
    print(f"   Incorrect predictions - Mean: {np.mean(incorrect_confidences):.3f}, Median: {np.median(incorrect_confidences):.3f}")


def visualize_tsne(model, x_test, y_test, class_names, n_samples=1000, 
                   save_path='images/tsne_visualization.png'):
    """
    Create t-SNE visualization of learned features from dense layer.
    
    Parameters:
    -----------
    model : keras.Model
        Trained CNN model
    x_test : numpy.ndarray
        Test images
    y_test : numpy.ndarray
        Test labels
    class_names : list
        List of class names
    n_samples : int
        Number of samples to use (t-SNE is slow on large datasets)
    save_path : str
        Path to save the visualization
    """
    print(f"\n📊 Generating t-SNE visualization (using {n_samples} samples)...")
    print("   This may take 1-2 minutes...")
    
    # Ensure model has been called at least once
    _ = model.predict(x_test[0:1], verbose=0)
    
    # Find the last dense layer before output
    dense_layer = None
    for layer in reversed(model.layers):
        if 'dense' in layer.name.lower() and layer.name != 'output':
            dense_layer = layer
            break
    
    if dense_layer is None:
        print("❌ Could not find dense layer for feature extraction")
        return
    
    # Create feature extractor
    feature_extractor = Model(inputs=model.input, outputs=dense_layer.output)
    
    # Extract features from subset of test data
    subset_size = min(n_samples, len(x_test))
    features = feature_extractor.predict(x_test[:subset_size], verbose=0)
    labels = y_test[:subset_size].flatten() if len(y_test.shape) > 1 else y_test[:subset_size]
    
    # Apply t-SNE
    print("   Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(features)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Plot each class with different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = (labels == class_idx)
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[color], label=class_name, alpha=0.6, s=25, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
    ax.set_title(f't-SNE Visualization of Learned Features\n({dense_layer.name} layer, {subset_size} samples)', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.show()


def generate_all_visualizations(model, x_test, y_test, class_names):
    """
    Generate all bonus visualizations.
    
    Parameters:
    -----------
    model : keras.Model
        Trained CNN model
    x_test : numpy.ndarray
        Test images (preprocessed/normalized)
    y_test : numpy.ndarray
        Test labels
    class_names : list
        List of class names
    """
    print("="*60)
    print("GENERATING ALL BONUS VISUALIZATIONS")
    print("="*60)
    
    create_images_directory()
    
    try:
        visualize_conv1_filters(model)
    except Exception as e:
        print(f"❌ Error generating Conv1 filters: {e}")
    
    try:
        visualize_feature_maps(model, x_test)
    except Exception as e:
        print(f"❌ Error generating feature maps: {e}")
    
    try:
        visualize_confidence_distribution(model, x_test, y_test)
    except Exception as e:
        print(f"❌ Error generating confidence distribution: {e}")
    
    try:
        visualize_tsne(model, x_test, y_test, class_names, n_samples=1000)
    except Exception as e:
        print(f"❌ Error generating t-SNE visualization: {e}")
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nCheck the 'images/' directory for all generated visualizations.")


# Example usage (if run directly in notebook):
if __name__ == "__main__":
    print(__doc__)
    print("\nTo use this script, import it in your notebook after training:")
    print("\n    from generate_bonus_visualizations import generate_all_visualizations")
    print("    generate_all_visualizations(cnn_model, x_test_normalized, y_test, class_names)\n")
