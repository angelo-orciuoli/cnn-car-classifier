"""
Main training script for car image classification.

This script trains multiple CNN models for classifying car images into 5 categories:
- SUV, Sedan, Pickup_Truck, Hatchback, Sports_Car
"""

import os
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import (
    setup_data_directories, split_data, augment_training_data,
    create_datasets, prepare_data_from_source, print_class_distribution
)
from models import CarClassifierModels, train_model, evaluate_model
from utils import save_model, display_training_history, create_confusion_matrix


def main():
    """Main training pipeline."""
    
    print("=== Car Image Classification Training Pipeline ===\n")
    
    # Configuration
    CAR_TYPES = ['SUV', 'Sedan', 'Pickup_Truck', 'Hatchback', 'Sports_Car']
    IMAGE_SIZE = (180, 180)
    BATCH_SIZE = 32
    EPOCHS = 7
    EPOCHS_EXTENDED = 15  # For models 4 and 5
    
    # Data augmentation targets to balance classes
    AUGMENTATION_TARGETS = {
        'Hatchback': 971,
        'SUV': 207,
        'Pickup_Truck': 31,
        'Sedan': 478,
        'Sports_Car': 974
    }

    
    # Step 1: Setup directory structure
    print("1. Setting up data directories...")
    setup_data_directories(".", CAR_TYPES)
    
    # Step 2: Split data into train/validation/test
    print("2. Splitting data into train/validation/test sets...")
    split_data(CAR_TYPES)
    
    # Step 3: Apply data augmentation
    print("3. Applying data augmentation to balance classes...")
    augment_training_data(CAR_TYPES, AUGMENTATION_TARGETS)
    
    # Step 4: Print data distribution
    print_class_distribution("train", "validation", "test")
    
    # Step 5: Create datasets
    print("4. Creating TensorFlow datasets...")
    train_ds, val_ds, test_ds = create_datasets(IMAGE_SIZE, BATCH_SIZE)
    
    # Step 6: Initialize model builder
    model_builder = CarClassifierModels(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=len(CAR_TYPES)
    )
    
    # Step 7: Train and evaluate all models
    results = {}
    
    # Model 1: VGG16 Transfer Learning
    print("\n" + "="*50)
    print("TRAINING MODEL 1: VGG16 Transfer Learning")
    print("="*50)
    model1 = model_builder.build_vgg16_model1()
    history1 = train_model(model1, train_ds, val_ds, EPOCHS, "Model 1 (VGG16)")
    loss1, acc1 = evaluate_model(model1, test_ds, "Model 1 (VGG16)")
    results['Model 1'] = {'loss': loss1, 'accuracy': acc1}
    save_model(model1, "model1_vgg16")
    display_training_history(history1, "Model 1 (VGG16)")
    
    # Model 2: Enhanced VGG16 (Best performing)
    print("\n" + "="*50)
    print("TRAINING MODEL 2: Enhanced VGG16 with ImageNet Classes")
    print("="*50)
    model2 = model_builder.build_vgg16_model2()
    history2 = train_model(model2, train_ds, val_ds, EPOCHS, "Model 2 (VGG16 Enhanced)")
    loss2, acc2 = evaluate_model(model2, test_ds, "Model 2 (VGG16 Enhanced)")
    results['Model 2'] = {'loss': loss2, 'accuracy': acc2}
    save_model(model2, "model2_vgg16_enhanced")
    display_training_history(history2, "Model 2 (VGG16 Enhanced)")
    
    # Model 3: Custom CNN
    print("\n" + "="*50)
    print("TRAINING MODEL 3: Custom 15-layer CNN")
    print("="*50)
    model3 = model_builder.build_custom_cnn_model3()
    history3 = train_model(model3, train_ds, val_ds, EPOCHS, "Model 3 (Custom CNN)")
    loss3, acc3 = evaluate_model(model3, test_ds, "Model 3 (Custom CNN)")
    results['Model 3'] = {'loss': loss3, 'accuracy': acc3}
    save_model(model3, "model3_custom_cnn")
    display_training_history(history3, "Model 3 (Custom CNN)")
    
    # Model 4: Lightweight CNN
    print("\n" + "="*50)
    print("TRAINING MODEL 4: Lightweight CNN")
    print("="*50)
    model4 = model_builder.build_lightweight_cnn_model4()
    history4 = train_model(model4, train_ds, val_ds, EPOCHS_EXTENDED, "Model 4 (Lightweight)")
    loss4, acc4 = evaluate_model(model4, test_ds, "Model 4 (Lightweight)")
    results['Model 4'] = {'loss': loss4, 'accuracy': acc4}
    save_model(model4, "model4_lightweight")
    display_training_history(history4, "Model 4 (Lightweight)")
    
    # Model 5: Deep CNN
    print("\n" + "="*50)
    print("TRAINING MODEL 5: Deep CNN")
    print("="*50)
    model5 = model_builder.build_deep_cnn_model5()
    history5 = train_model(model5, train_ds, val_ds, EPOCHS_EXTENDED, "Model 5 (Deep CNN)")
    loss5, acc5 = evaluate_model(model5, test_ds, "Model 5 (Deep CNN)")
    results['Model 5'] = {'loss': loss5, 'accuracy': acc5}
    save_model(model5, "model5_deep_cnn")
    display_training_history(history5, "Model 5 (Deep CNN)")
    
    # Step 8: Print final results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:<15}% {metrics['loss']:<15}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest performing model: {best_model[0]} with {best_model[1]['accuracy']}% accuracy")
    
    # Step 9: Generate confusion matrix for best model (Model 2)
    print(f"\nGenerating confusion matrix for {best_model[0]}...")
    if best_model[0] == 'Model 2':
        create_confusion_matrix(model2, test_ds, CAR_TYPES)
    
    print("\n Training completed successfully!")
    print("\nNext steps:")
    print("1. Use 'CarImageClassifierTest.ipynb' to test Model 2 with external images")
    print("2. All trained models are saved in the 'saved_models/' directory")
    print("3. Model 2 (Enhanced VGG16) is recommended for production use")


if __name__ == "__main__":
    main()
