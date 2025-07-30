# Car Image Classifier

A deep learning project that classifies car images into 5 different categories using Convolutional Neural Networks (CNNs).

## Overview

This project implements a car image classification system capable of distinguishing between:
- **Sedans**
- **SUVs** 
- **Hatchbacks**
- **Sports Cars**
- **Pickup Trucks**

The classifier leverages transfer learning with VGG16 architecture and custom CNN models to achieve high accuracy in automotive image recognition.

## Motivation

With the rapid growth of autonomous vehicles and automotive AI applications, accurate vehicle classification has become increasingly important. This project explores the effectiveness of different CNN architectures for distinguishing between similar vehicle types, particularly challenging cases like sedans vs. hatchbacks.

## Technical Approach

### Dataset
- **Source**: Kaggle automotive image dataset
- **Format**: 180x180 JPEG images
- **Distribution**: ~6,500 images across 5 vehicle categories
- **Split**: 75% training, 12.5% validation, 12.5% testing

### Data Augmentation
To address class imbalance and improve model robustness, I implemented:
- Horizontal flipping
- Random rotation (±10 degrees)
- Gaussian blur and noise injection
- Brightness and contrast normalization

### Model Architectures

**Model 1 & 2: VGG16 Transfer Learning**
- Pre-trained VGG16 backbone with frozen weights
- Custom classification head with dropout regularization
- Model 2 incorporates ImageNet's pre-trained 'sports_car' and 'pickup' classes
- 7 epochs training with Adam optimizer

**Model 3: Custom CNN**
- 15-layer architecture with 5 conv-pool blocks
- ReLU activation functions
- Dropout layers for regularization

**Model 4: Lightweight CNN**
- Simplified architecture for efficiency comparison
- Single dropout layer
- 15 epochs training

**Model 5: Deep CNN**
- Extended version of Model 3 with additional layers
- Deeper feature extraction capabilities

## Results

| Model | Test Accuracy | Test Loss | Architecture |
|-------|---------------|-----------|--------------|
| Model 1 | 95.6% | 0.14 | VGG16 Transfer Learning |
| **Model 2** | **97.3%** | **0.10** | **VGG16 + ImageNet Classes** |
| Model 3 | 85.1% | 0.44 | Custom 15-layer CNN |
| Model 4 | 91.5% | 0.64 | Lightweight CNN |
| Model 5 | 94.2% | 0.36 | Deep CNN |

**Model 2** achieved the best performance, demonstrating the effectiveness of leveraging pre-trained features from ImageNet's vehicle categories.

## Key Findings

- Transfer learning significantly outperforms custom architectures
- VGG16's pre-trained features provide excellent feature representations
- Data augmentation effectively addresses class imbalance
- The model excels at distinguishing sports cars and pickup trucks
- Some bias toward vehicle types present in ImageNet training data

## Usage

### Training Models
```python
python main.py
```

### Testing with Custom Images
Use the Jupyter notebook `CarImageClassifierTest.ipynb` to test Model 2 with your own images.

## Applications

This classifier has practical applications in:
- **Automotive Industry**: Vehicle inventory management
- **Insurance**: Automated damage assessment
- **Traffic Monitoring**: Vehicle type analytics
- **E-commerce**: Automated vehicle listing categorization
- **Smart Cities**: Traffic flow analysis

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure
```
├── main.py                    # Main training script
├── models.py                  # Model definitions
├── data_preprocessing.py      # Data handling utilities
├── utils.py                   # Helper functions
├── requirements.txt           # Dependencies
└── External Images/           # Test images
```

## Future Improvements

- Implement real-time video classification
- Add more vehicle categories (motorcycles, buses, etc.)
- Experiment with modern architectures (ResNet, EfficientNet)
- Deploy as web service or mobile app
- Integrate with automotive APIs
