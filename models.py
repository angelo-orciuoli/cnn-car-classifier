"""
Model definitions for car image classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np


class CarClassifierModels:
    """Collection of different CNN architectures for car classification."""
    
    def __init__(self, input_shape=(180, 180, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_vgg16_model1(self):
        """
        Build VGG16 transfer learning model (Model 1).
        
        Returns:
            Model: Compiled Keras model
        """
        # Load pre-trained VGG16 without top layers
        base_model = VGG16(include_top=False, input_shape=self.input_shape)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification head
        flat = layers.Flatten()(base_model.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        cls = layers.Dense(128, activation='relu')(drop1)
        drop2 = layers.Dropout(0.5)(cls)
        output = layers.Dense(self.num_classes, activation='softmax')(drop2)
        
        # Create final model
        model = Model(inputs=base_model.inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_vgg16_model2(self):
        """
        Build enhanced VGG16 model with ImageNet class integration (Model 2).
        
        Returns:
            Model: Compiled Keras model
        """
        # Get ImageNet class labels
        model_vgg16 = VGG16(weights='imagenet', include_top=True)
        class_labels = decode_predictions(np.zeros((1, 1000)), top=1000)
        class_names = [label[1] for label in class_labels[0]]
        
        # Find indices of relevant classes - use flexible search
        pickup_index = None
        sports_car_index = None
        
        # Search for pickup-related classes
        for i, name in enumerate(class_names):
            if 'pickup' in name.lower() or 'truck' in name.lower():
                pickup_index = i
                break
        
        # Search for sports car-related classes
        for i, name in enumerate(class_names):
            if 'sports' in name.lower() and 'car' in name.lower():
                sports_car_index = i
                break
        
        # If not found, use default indices
        if pickup_index is None:
            pickup_index = 0  # Default fallback
            print("Warning: No pickup class found in ImageNet, using default index")
        if sports_car_index is None:
            sports_car_index = 1  # Default fallback
            print("Warning: No sports car class found in ImageNet, using default index")
        
        # Create base model
        model_base = VGG16(include_top=False, input_shape=self.input_shape)
        
        # Freeze base model layers
        for layer in model_base.layers:
            layer.trainable = False
        
        # Add custom classification head
        flat = layers.Flatten()(model_base.layers[-1].output)
        drop1 = layers.Dropout(0.5)(flat)
        cls = layers.Dense(128, activation='relu')(drop1)
        drop2 = layers.Dropout(0.5)(cls)
        
        # Create extended class list with custom categories
        new_class_names = (class_names[:pickup_index] + 
                          ['Pickup_Truck'] + 
                          class_names[pickup_index+1:sports_car_index] + 
                          ['Sports_Car'] + 
                          class_names[sports_car_index+1:])
        
        output = layers.Dense(len(new_class_names), activation='softmax')(drop2)
        
        # Create final model
        model = Model(inputs=model_base.inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_custom_cnn_model3(self):
        """
        Build custom 15-layer CNN (Model 3).
        
        Returns:
            Model: Compiled Keras model
        """
        model = Sequential()
        
        # Convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                        input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Classification layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        return model
    
    def build_lightweight_cnn_model4(self):
        """
        Build lightweight CNN for efficiency comparison (Model 4).
        
        Returns:
            Model: Compiled Keras model
        """
        model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        
        # Compile model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        return model
    
    def build_deep_cnn_model5(self):
        """
        Build deep CNN with extended architecture (Model 5).
        
        Returns:
            Model: Compiled Keras model
        """
        model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(512, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        
        # Compile model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        return model
    
    def get_all_models(self):
        """
        Get all models as a dictionary.
        
        Returns:
            dict: Dictionary of model names and model objects
        """
        return {
            'Model_1_VGG16': self.build_vgg16_model1(),
            'Model_2_VGG16_Enhanced': self.build_vgg16_model2(),
            'Model_3_Custom_CNN': self.build_custom_cnn_model3(),
            'Model_4_Lightweight': self.build_lightweight_cnn_model4(),
            'Model_5_Deep_CNN': self.build_deep_cnn_model5()
        }


def train_model(model, train_ds, val_ds, epochs=7, model_name="Model"):
    """
    Train a model and return training history.
    
    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs (int): Number of training epochs
        model_name (str): Name for logging
    
    Returns:
        History: Training history
    """
    print(f"\n=== Training {model_name} ===")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1
    )
    return history


def evaluate_model(model, test_ds, model_name="Model"):
    """
    Evaluate model on test dataset.
    
    Args:
        model: Keras model to evaluate
        test_ds: Test dataset
        model_name (str): Name for logging
    
    Returns:
        tuple: (loss, accuracy)
    """
    print(f"\n=== Evaluating {model_name} ===")
    loss, accuracy = model.evaluate(test_ds)
    loss = round(loss, 2)
    accuracy = round(accuracy * 100, 2)
    
    print(f"{model_name} - Test Loss: {loss}")
    print(f"{model_name} - Test Accuracy: {accuracy}%")
    
    return loss, accuracy
