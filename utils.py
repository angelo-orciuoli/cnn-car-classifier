"""
Utility functions for car image classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def preprocess_image(image_path, target_size=(180, 180)):
    """
    Preprocess a single image for model prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size (width, height)
    
    Returns:
        np.array: Preprocessed image array ready for prediction
    """
    # Open and resize image
    img = Image.open(image_path)
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_image(model, image_path, target_size=(180, 180), class_names=None):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image file
        target_size (tuple): Target image size
        class_names (list): List of class names
    
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    if class_names is None:
        class_names = ['SUV', 'Sedan', 'Pickup_Truck', 'Hatchback', 'Sports_Car']
    
    # Preprocess image
    img = preprocess_image(image_path, target_size)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    
    # Get predicted class
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]
    
    return predicted_class, confidence_score


def visualize_prediction(image_path, predicted_class, confidence_score):
    """
    Display image with prediction results.
    
    Args:
        image_path (str): Path to the image file
        predicted_class (str): Predicted class name
        confidence_score (float): Confidence score
    """
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence_score:.3f})")
    plt.show()


def batch_predict_images(model, image_paths, target_size=(180, 180), class_names=None):
    """
    Predict classes for multiple images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image file paths
        target_size (tuple): Target image size
        class_names (list): List of class names
    
    Returns:
        list: List of tuples (image_path, predicted_class, confidence_score)
    """
    if class_names is None:
        class_names = ['SUV', 'Sedan', 'Pickup_Truck', 'Hatchback', 'Sports_Car']
    
    results = []
    
    for image_path in image_paths:
        try:
            predicted_class, confidence_score = predict_image(
                model, image_path, target_size, class_names
            )
            results.append((image_path, predicted_class, confidence_score))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((image_path, "Error", 0.0))
    
    return results


def save_model(model, model_name, save_dir="saved_models"):
    """
    Save a trained model to disk.
    
    Args:
        model: Keras model to save
        model_name (str): Name for the saved model
        save_dir (str): Directory to save models
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_name, save_dir="saved_models"):
    """
    Load a saved model from disk.
    
    Args:
        model_name (str): Name of the saved model
        save_dir (str): Directory containing saved models
    
    Returns:
        Model: Loaded Keras model
    """
    model_path = os.path.join(save_dir, f"{model_name}.h5")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def display_training_history(history, model_name="Model"):
    """
    Display training history plots.
    
    Args:
        history: Keras training history object
        model_name (str): Name of the model for plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def get_class_distribution(dataset_path):
    """
    Get the distribution of classes in a dataset directory.
    
    Args:
        dataset_path (str): Path to dataset directory
    
    Returns:
        dict: Dictionary with class names and counts
    """
    distribution = {}
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            distribution[class_name] = count
    
    return distribution


def print_class_distribution(train_path, val_path, test_path):
    """
    Print class distribution across train/validation/test sets.
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        test_path (str): Path to test data
    """
    train_dist = get_class_distribution(train_path)
    val_dist = get_class_distribution(val_path)
    test_dist = get_class_distribution(test_path)
    
    print("\n=== Dataset Distribution ===")
    print(f"{'Class':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 50)
    
    for class_name in train_dist.keys():
        train_count = train_dist.get(class_name, 0)
        val_count = val_dist.get(class_name, 0)
        test_count = test_dist.get(class_name, 0)
        total_count = train_count + val_count + test_count
        
        print(f"{class_name:<15} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    print("-" * 50)
    total_train = sum(train_dist.values())
    total_val = sum(val_dist.values())
    total_test = sum(test_dist.values())
    total_all = total_train + total_val + total_test
    
    print(f"{'Total':<15} {total_train:<8} {total_val:<8} {total_test:<8} {total_all:<8}")


def create_confusion_matrix(model, test_ds, class_names=None):
    """
    Create and display confusion matrix for model predictions.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        class_names (list): List of class names
    """
    if class_names is None:
        class_names = ['SUV', 'Sedan', 'Pickup_Truck', 'Hatchback', 'Sports_Car']
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        # Get predictions
        y_pred = model.predict(test_ds, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Get true labels
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Print classification report
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred_classes, target_names=class_names))
        
    except ImportError:
        print("sklearn and seaborn required for confusion matrix. Install with: pip install scikit-learn seaborn")