"""
Data preprocessing and augmentation utilities for car image classification.
"""

import os
import random
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
import Augmentor
import imgaug.augmenters as iaa


def delete_non_jpeg_images(directory):
    """
    Remove all non-JPEG images from a directory.
    
    Args:
        directory (str): Path to directory to clean
    
    Returns:
        list: List of deleted file names
    """
    deleted_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            with Image.open(filepath) as img:
                if img.format != 'JPEG':
                    os.remove(filepath)
                    print(f"Deleted: {filename}")
                    deleted_files.append(filename)
        except:
            os.remove(filepath)
            deleted_files.append(filename)
    return deleted_files


def augment_images(images):
    """
    Apply data augmentation to a batch of images using imgaug.
    
    Args:
        images (list): List of image arrays
    
    Returns:
        list: List of augmented image arrays
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-10, 10)),  # random rotations
        iaa.GaussianBlur(sigma=(0, 1.0)),  # blur images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # add Gaussian noise
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # multiply brightness
        iaa.ContrastNormalization((0.8, 1.2))  # contrast normalization
    ], random_order=True)
    
    augmented_images = seq(images=images)
    return augmented_images


def read_and_augment_images(folder_path):
    """
    Read images from folder and apply augmentation.
    
    Args:
        folder_path (str): Path to folder containing images
    
    Returns:
        list: List of augmented image arrays
    """
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            img_array = np.array(img)
            images.append(img_array)
    augmented_images = augment_images(images)
    return augmented_images


def move_files(source_dir, files, destination_dir):
    """
    Move files from source to destination directory.
    
    Args:
        source_dir (str): Source directory path
        files (list): List of file names to move
        destination_dir (str): Destination directory path
    """
    for file_name in files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.move(source_path, destination_path)


def setup_data_directories(base_path, car_types):
    """
    Setup train/validation/test directory structure.
    
    Args:
        base_path (str): Base path for data
        car_types (list): List of car type names
    """
    directories = ['train', 'validation', 'test']
    
    # Create main directories
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    # Create subdirectories for each car type
    for directory in directories:
        for car_type in car_types:
            subdir = os.path.join(directory, car_type)
            os.makedirs(subdir, exist_ok=True)


def split_data(car_types, num_validation=200, num_test=200):
    """
    Split data into train/validation/test sets.
    
    Args:
        car_types (list): List of car type directory names
        num_validation (int): Number of validation images per class
        num_test (int): Number of test images per class
    """
    for car_type in car_types:
        # List all files in the current directory
        files = os.listdir(car_type)
        random.shuffle(files)
        
        # Calculate splits
        num_train_files = len(files) - num_validation - num_test
        
        train_files = files[:num_train_files]
        validation_files = files[num_train_files:num_train_files + num_validation]
        test_files = files[num_train_files + num_validation:]
        
        # Move files to respective directories
        train_subdir = os.path.join('train', car_type)
        validation_subdir = os.path.join('validation', car_type)
        test_subdir = os.path.join('test', car_type)
        
        move_files(car_type, train_files, train_subdir)
        move_files(car_type, validation_files, validation_subdir)
        move_files(car_type, test_files, test_subdir)
    
    print("Data split completed successfully!")


def augment_training_data(car_types, target_samples):
    """
    Apply data augmentation to balance training data across car types.
    
    Args:
        car_types (list): List of car type names
        target_samples (dict): Target number of samples per car type
    """
    for car_type in car_types:
        train_directory = f"train/{car_type}"
        
        if car_type not in target_samples:
            continue
            
        # Create augmentation pipeline
        p = Augmentor.Pipeline(train_directory)
        p.flip_left_right(probability=0.5)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        
        # Generate augmented samples
        p.sample(target_samples[car_type])
        
        # Move generated files from output directory
        output_dir = os.path.join(train_directory, "output")
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                src = os.path.join(output_dir, filename)
                dst = os.path.join(train_directory, filename)
                shutil.move(src, dst)
            os.rmdir(output_dir)


def create_datasets(image_size=(180, 180), batch_size=32):
    """
    Create TensorFlow datasets for training, validation, and testing.
    
    Args:
        image_size (tuple): Target image size
        batch_size (int): Batch size for datasets
    
    Returns:
        tuple: (train_ds, val_ds, test_ds)
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "train",
        labels='inferred',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "validation",
        labels='inferred',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "test",
        labels='inferred',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return train_ds, val_ds, test_ds


def prepare_data_from_source(data_folder_path, car_types):
    """
    Complete data preparation pipeline from source folder.
    
    Args:
        data_folder_path (str): Path to source data folder
        car_types (list): List of car type names
    """
    for car_type in car_types:
        car_type_folder_path = os.path.join(data_folder_path, car_type)
        
        if os.path.exists(car_type_folder_path):
            target_directory = car_type
            
            # Create target directory
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            
            # Clean non-JPEG files
            delete_non_jpeg_images(car_type_folder_path)
            
            # Copy and rename files
            files_after_deletion = os.listdir(car_type_folder_path)
            for i, file_name in enumerate(files_after_deletion):
                new_file_name = f"{car_type}_{i+1}.jpg"
                current_file_path = os.path.join(car_type_folder_path, file_name)
                new_file_path = os.path.join(target_directory, new_file_name)
                shutil.copy(current_file_path, new_file_path)
    
    print("Data preparation completed!")