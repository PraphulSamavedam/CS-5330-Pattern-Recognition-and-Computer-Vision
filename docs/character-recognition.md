# Character Recognition with Deep Learning

[← Back to Home](index.md)

## Overview

This project implements a **Character Recognition System** using deep Convolutional Neural Networks (CNNs) built with PyTorch. The system recognizes handwritten digits from the MNIST dataset and extends to recognize Greek letters through transfer learning. The project explores various CNN architectures, training strategies, and visualization techniques to understand how deep neural networks learn to recognize characters.

The system achieves **98%+ accuracy on MNIST digit recognition** and successfully transfers learned features to recognize Greek letters (**90%+ accuracy**) with minimal additional training.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CHARACTER RECOGNITION WITH DEEP LEARNING                  │
│                              System Workflow Diagram                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │     BASE TRAINING PHASE          │
                    │  (MNIST Digit Recognition)       │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ MNIST Dataset (70K images)   │
                    │ - Train: 60,000 images       │
                    │ - Test: 10,000 images        │
                    │ - Size: 28×28 grayscale      │
                    │ - Classes: 0-9 (10 digits)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Neural Network Architecture │
                    │  (BaseNetwork - LeNet-5)     │
                    │  ┌────────────────────────┐  │
                    │  │ Conv1: 1→10, 5×5       │  │
                    │  │ MaxPool: 2×2           │  │
                    │  │ ReLU                   │  │
                    │  ├────────────────────────┤  │
                    │  │ Conv2: 10→20, 5×5      │  │
                    │  │ Dropout2d: p=0.5       │  │
                    │  │ MaxPool: 2×2           │  │
                    │  │ ReLU                   │  │
                    │  ├────────────────────────┤  │
                    │  │ Flatten: 320 features  │  │
                    │  ├────────────────────────┤  │
                    │  │ FC1: 320→50            │  │
                    │  │ ReLU                   │  │
                    │  │ FC2: 50→10             │  │
                    │  │ LogSoftmax             │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Training Loop (5 Epochs)    │
                    │  ├─ Forward Pass             │
                    │  ├─ Compute Loss (NLLLoss)   │
                    │  ├─ Backward Pass            │
                    │  ├─ Update Weights (SGD)     │
                    │  │   • Learning Rate: 0.01   │
                    │  │   • Momentum: 0.5         │
                    │  │   • Batch Size: 64        │
                    │  └─ Log Progress             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Save Trained Model          │
                    │  models/final_model.pth      │
                    │  • Model weights             │
                    │  • Test accuracy: 98%+       │
                    └──────────────────────────────┘
                                   │
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                                                      │
        ▼                                                      ▼
┌────────────────────┐                           ┌────────────────────┐
│ TRANSFER LEARNING  │                           │  EXPLORATION PHASE │
│ (Greek Letters)    │                           │  (Architecture)    │
└────────┬───────────┘                           └────────┬───────────┘
         │                                                │
         ▼                                                ▼
┌────────────────────────────────┐              ┌────────────────────┐
│  Load Pretrained Model         │              │ Network Variations │
│  models/final_model.pth        │              │ ├─ Kernel Size     │
└────────┬───────────────────────┘              │ ├─ Filter Count    │
         │                                      │ ├─ Network Depth   │
         ▼                                      │ ├─ Dropout Rate    │
┌────────────────────────────────┐              │ ├─ Batch Size      │
│  Freeze Convolutional Layers   │              │ └─ Optimizers      │
│  • Conv1, Conv2: frozen        │              └────────────────────┘
│  • FC layers: trainable        │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Modify Last Layer             │
│  • 50→10 becomes 50→3          │
│  • For 3 Greek letters         │
│  • OR 50→7 for 7 letters       │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Greek Dataset                 │
│  • Alpha, Beta, Gamma          │
│  • 27 images per class         │
│  • Transformed to 28×28        │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Fine-tune Training            │
│  • 15-45 epochs                │
│  • Small learning rate (0.03)  │
│  • Small batch size (5-6)      │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Save Greek Model              │
│  models/model_greek.pth        │
│  models/model_extended_greek   │
└────────────────────────────────┘

Key Algorithms:
├─ Convolutional Neural Networks: Feature extraction
├─ Transfer Learning: Reuse learned features
├─ Stochastic Gradient Descent: Weight optimization
├─ Backpropagation: Gradient computation
├─ Dropout: Regularization to prevent overfitting
└─ Data Augmentation: GreekTransform for preprocessing
```

---

## Key Features

### Core Capabilities

- **MNIST Digit Recognition** - Train CNN from scratch (98%+ accuracy)
- **Transfer Learning** - Adapt pre-trained model to Greek letters (90%+ accuracy)
- **Network Architecture Exploration** - Compare different CNN designs and hyperparameters
- **Filter Visualization** - Understand what convolutional filters learn
- **Custom Image Testing** - Test on user-provided images via file dialog
- **Training Monitoring** - Real-time loss and accuracy visualization

### Neural Network Architectures

The system implements multiple CNN architectures:

1. **BaseNetwork (LeNet-5 inspired)** - Standard architecture with 10 and 20 filters
   - Conv1: 1→10 filters, 5×5 kernel
   - MaxPool: 2×2
   - Conv2: 10→20 filters, 5×5 kernel
   - Dropout2d: p=0.5
   - MaxPool: 2×2
   - FC1: 320→50
   - FC2: 50→10 (or 3/7 for Greek)
   - Total Parameters: 21,840

2. **NetWorkKernel1** - Uses 1×1 kernels instead of 5×5
3. **ManyParallelFiltersNetWork** - Configurable filter counts
4. **DeepNetwork1** - Adds one extra convolution per layer
5. **DeepNetwork2** - Adds two extra convolutions per layer

### Transfer Learning Approach

**Feature Extraction Strategy:**
1. Freeze pretrained convolutional layers (Conv1, Conv2)
2. Replace final layer (10→3 or 10→7 classes)
3. Fine-tune only the new final layer
4. Requires minimal training data (27 images per class)

**Benefits:**
- 99% reduction in trainable parameters
- Trains in ~45 seconds (vs. hours from scratch)
- Works with limited data (81 images for 3 classes)
- Leverages edge/curve/shape features from MNIST

---

## Technical Details

### CNN Architecture (BaseNetwork)

```
Input: 28×28×1 grayscale image

Layer 1: Convolutional Feature Extraction
├─ Conv2d(1→10, kernel=5×5) → 24×24×10 (edge detectors)
├─ MaxPool2d(2×2, stride=2) → 12×12×10 (downsample)
└─ ReLU() → Non-linearity

Layer 2: Higher-Level Features
├─ Conv2d(10→20, kernel=5×5) → 8×8×20 (combined features)
├─ Dropout2d(p=0.5) → Regularization
├─ MaxPool2d(2×2, stride=2) → 4×4×20
└─ ReLU()

Flatten: 4×4×20 → 320-dimensional vector

Layer 3: Classification
├─ Linear(320→50) → Abstract representations
├─ ReLU()
├─ Linear(50→10) → Class scores
└─ LogSoftmax() → Log probabilities

Total Parameters: 21,840
```

### Training Algorithm

**Forward Pass:**
```
Input [B, 1, 28, 28] → Conv1 → Pool → ReLU →
Conv2 → Dropout → Pool → ReLU → Flatten →
FC1 → ReLU → FC2 → LogSoftmax → Output [B, 10]
```

**Backward Pass (SGD with Momentum):**
```
1. Compute loss: NLLLoss(predictions, labels)
2. Compute gradients: loss.backward()
3. Update weights: velocity = μ×velocity - η×gradient
                  weight = weight + velocity

Where:
  η = learning rate (0.01)
  μ = momentum (0.5)
```

**Optimization Details:**
- **Learning Rate**: 0.01 (MNIST), 0.03 (Greek transfer)
- **Momentum**: 0.5 (accelerates convergence)
- **Batch Size**: 64 (MNIST), 5-6 (Greek)
- **Loss Function**: NLLLoss (Negative Log Likelihood)
- **Regularization**: Dropout (p=0.5)

### Transfer Learning Process

```python
# Step 1: Load pretrained model
model = BaseNetwork()
model.load_state_dict(torch.load("models/final_model.pth"))

# Step 2: Freeze convolutional layers
for param in model.convolution_stack.parameters():
    param.requires_grad = False

# Step 3: Replace final layer
model.classification_stack[-2] = nn.Linear(50, num_classes)

# Step 4: Fine-tune (train only final layer)
optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.03, momentum=0.1)
```

### GreekTransform Preprocessing

Converts Greek letter images (128×128 RGB) to MNIST format (28×28 grayscale):

```
1. RGB → Grayscale: weighted sum (0.299×R + 0.587×G + 0.114×B)
2. Affine Scaling: scale factor 36/128
3. Center Crop: crop to 28×28 from center
4. Invert Colors: 255 - pixel (white-on-black → black-on-white)
5. Normalize: (pixel/255 - 0.1307) / 0.3801
```

---

## Benefits and Limitations

!!! success "Benefits"
    - **High Accuracy**: 98%+ on MNIST, 90%+ on Greek letters
    - **Fast Training**: 5 epochs (~10 minutes) for MNIST base model
    - **Transfer Learning**: Achieves 90%+ with only 27 images per class
    - **Minimal Data**: Greek model trained with 81 total images
    - **Interpretable**: Filter visualization shows learned features
    - **Extensible**: Easy to add new architectures and experiments
    - **Well-Documented**: Comprehensive code with visualization tools

!!! warning "Limitations"
    - **Fixed Input Size**: Requires 28×28 images
    - **Grayscale Only**: No color information used
    - **Limited Augmentation**: Basic transformations only
    - **No Real-time**: Not optimized for live video
    - **CPU Training**: No GPU optimization in code
    - **Single Font**: Greek letters must match training style
    - **Memory Usage**: Larger batch sizes require significant RAM

---

## Usage

### 1. Training on MNIST Digits

Train the base CNN model on MNIST:

```bash
python src/train_basic.py
```

**Default hyperparameters:**
- Epochs: 5
- Learning rate: 0.01
- Momentum: 0.5
- Batch size: 64

**Custom hyperparameters:**
```bash
python src/train_basic.py -e 10 -r 0.02 -m 0.9 -br 128
```

**Command-line arguments:**
- `-e, --epochs` - Number of training epochs
- `-r, --rate` - Learning rate
- `-m, --momentum` - SGD momentum
- `-br, --train_batch_size` - Training batch size

**Output:**
- `models/final_model.pth` - Trained model weights
- `base_network.png` - Network architecture visualization
- Training curves showing loss vs. examples seen

### 2. Testing the Trained Model

Evaluate the trained model on MNIST test data:

```bash
python src/test_basic.py
```

**Output:**
- Visual display of predictions on 9 random test samples
- Console output with prediction accuracy

### 3. Transfer Learning (3 Greek Letters)

Adapt the MNIST model to recognize Alpha, Beta, Gamma:

```bash
python src/transfer_greek.py
```

**Parameters:**
- Epochs: 15
- Learning rate: 0.03
- Batch size: 5
- Classes: 3 (α, β, γ)

**Output:**
- `models/model_greek.pth` - Transfer learned model
- `greek_network.png` - Modified architecture
- Training accuracy plots

### 4. Transfer Learning (7 Greek Letters)

Recognize 7 Greek letter classes:

```bash
python src/extended_greek.py
```

**Parameters:**
- Epochs: 45
- Classes: 7 (α, β, γ, η, δ, θ, φ)

**Output:**
- `models/model_extended_greek.pth` - Extended model
- Accuracy and loss curves

### 5. Testing Greek Models

```bash
# Test 3-letter model
python src/transfer_greek_testing.py

# Test 7-letter model
python src/extended_greek_testing.py
```

### 6. Custom Image Testing

Test on your own handwritten digit images:

```bash
python src/user_testing.py
```

**What it does:**
1. Opens file dialog for image selection
2. Allows multiple image selection
3. Preprocesses (resize, grayscale, normalize)
4. Predicts digit for each image
5. Displays results with accuracy

**Image requirements:**
- Any size (will be resized to 28×28)
- PNG, JPG, or other PIL-supported formats
- Filename: `<digit>_description.png` (e.g., `5_myhandwriting.png`)

### 7. Network Architecture Exploration

Explore different architectures and hyperparameters:

```bash
python src/explore_cnn.py
```

**Experiments:**
- Filter size variation (1×1, 5×5, 9×9)
- Network depth comparison
- Filter count variation

### 8. Extensions and Advanced Experiments

```bash
python src/extensions.py
```

**Experiments:**
- Batch size variation (32, 64, 128, 512)
- Dropout rate variation (0.0-1.0)
- Optimizer comparison (SGD, Adam, Adagrad, Adadelta)
- Fashion-MNIST transfer learning

### 9. Filter Visualization

Visualize learned convolutional filters:

```bash
python src/visualize_layers.py
```

**Output:**
- `results/layer_0_filters.png` - Visualization of 10 Conv1 filters
- `results/Information learned in layer 0.png` - Filter responses
- `results/model_weights.log` - Detailed filter weights

---

## Performance Metrics

### MNIST Digit Recognition

| Metric | Value |
|--------|-------|
| Training Accuracy | 98-99% |
| Test Accuracy | 98-99% |
| Training Time (5 epochs) | ~5-10 minutes (CPU) |
| Reprojection Error | N/A |
| Model Size | ~90KB |

### Greek Letter Recognition (Transfer Learning)

| Dataset | Classes | Training Images | Accuracy | Training Time |
|---------|---------|----------------|----------|---------------|
| 3 Letters | α, β, γ | 81 (27 each) | 90-92% | ~45 seconds |
| 7 Letters | +η, δ, θ, φ | ~189 | 85-95% | ~2 minutes |

### Architecture Comparison

| Network | Kernel | Depth | Parameters | Accuracy | Time/Epoch |
|---------|--------|-------|------------|----------|------------|
| BaseNetwork | 5×5 | 2 conv | 21,840 | 98.5% | ~2 min |
| Kernel1 | 1×1 | 2 conv | 54,510 | 96.0% | ~1.5 min |
| DeepNetwork1 | 5×5 | 4 conv | 24,360 | 98.8% | ~3 min |
| DeepNetwork2 | 5×5 | 6 conv | 26,880 | 98.9% | ~4 min |

**Insight:** BaseNetwork offers best speed/accuracy trade-off for MNIST.

---

## Learning Objectives

This project demonstrates understanding of:

1. **Deep Learning Fundamentals** - Neural network architectures, forward/backward propagation
2. **Convolutional Neural Networks** - Conv layers, pooling, feature extraction
3. **Training Strategies** - SGD with momentum, learning rate, batch size
4. **Regularization** - Dropout for preventing overfitting
5. **Transfer Learning** - Feature extraction, layer freezing, fine-tuning
6. **Loss Functions** - NLLLoss, LogSoftmax
7. **Data Preprocessing** - Normalization, transformations
8. **Hyperparameter Tuning** - Grid search, performance comparison
9. **Model Evaluation** - Accuracy, confusion matrices, visualization
10. **PyTorch Framework** - Model definition, training loops, data loaders

---

## Installation

**Requirements:**
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- Pillow (PIL)
- torchviz
- OpenCV (optional)

**Quick install:**
```bash
pip install torch torchvision matplotlib pillow torchviz opencv-python
```

For detailed installation and troubleshooting, see the [Development Guide](https://github.com/praphul-kumar/CS-5330-Pattern-Recognition-and-Computer-Vision/blob/main/Assignment5/CharacterRecognition/DEVELOPMENT.md).

---

## References

- [LeNet-5 Paper (LeCun et al., 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Understanding CNNs (CS231n)](http://cs231n.github.io/convolutional-networks/)
- [Dropout Paper (Srivastava et al., 2014)](http://jmlr.org/papers/v15/srivastava14a.html)

---

[← Back to Home](index.md)
