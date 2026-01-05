# Real-Time 2D Object Recognition (RT2DOR)

## Overview

The Real-Time 2D Object Recognition (RT2DOR) system is a comprehensive computer vision solution that detects, segments, and classifies objects in both static images and live video streams. Built on classical computer vision techniques, the system combines image segmentation with K-Nearest Neighbors (K-NN) classification to recognize objects based on their geometric and spatial features.

The system recognizes **17 different object classes** and achieves **83% average accuracy** on the test dataset. It provides multiple executables for different use cases: training classifiers, analyzing static images, detecting multiple objects, generating confusion matrices, and performing real-time video recognition.

**Demo Video:** [Watch the system in action](https://youtu.be/ZORyRWK4H6c)

---

## System Architecture

The RT2DOR system follows a modular pipeline architecture with distinct training and recognition phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME 2D OBJECT RECOGNITION                          │
│                         System Workflow Diagram                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │     TRAINING PHASE               │
                    │  (Offline - One Time Setup)      │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ Training Images Folder       │
                    │ (Labeled: Beanie, Cup, etc.) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  For Each Training Image:    │
                    │  1. Threshold (Binary)       │
                    │  2. Morphology (Clean)       │
                    │  3. Region Growing (Segment) │
                    │  4. Extract Features         │
                    │     - Area Ratio             │
                    │     - Aspect Ratio           │
                    │     - Percent Filled         │
                    │     - Hu Moments (7 values)  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Store in features.csv       │
                    │  Format:                     │
                    │  path,label,feat1,...,feat9  │
                    └──────────────┬───────────────┘
                                   │
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                     │
         ▼                                                     ▼
┌────────────────────┐                            ┌────────────────────┐
│ RECOGNITION PHASE  │                            │  EVALUATION PHASE  │
│ (Real-time/Static) │                            │  (Offline)         │
└────────┬───────────┘                            └────────┬───────────┘
         │                                                  │
         ▼                                                  ▼
┌────────────────────┐                         ┌──────────────────────┐
│ Input Source:      │                         │ Confusion Matrix     │
│ - Live Video       │                         │ Generator            │
│ - Static Image     │                         │ ├─ Load features.csv │
│ - Multiple Objects │                         │ ├─ K-NN Predictions  │
└────────┬───────────┘                         │ ├─ Compare Labels    │
         │                                     │ └─ Generate Matrix   │
         ▼                                     └──────────────────────┘
┌──────────────────────────────┐
│  Image Processing Pipeline:  │
│  1. Threshold                │
│     └─ Binary Image          │
│  2. Dilation                 │
│     └─ Fill Holes            │
│  3. Erosion                  │
│     └─ Remove Noise          │
│  4. Region Growing           │
│     └─ Connected Components  │
│  5. Top N Segments           │
│     └─ Filter by Area        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Feature Extraction          │
│  For Each Detected Region:   │
│  ├─ Compute Moments          │
│  ├─ Calculate Dimensions     │
│  ├─ Extract Hu Moments       │
│  └─ Generate Feature Vector  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  K-NN Classification         │
│  1. Load features.csv        │
│  2. Compute Distances        │
│     - Euclidean (default)    │
│     - Scaled Euclidean       │
│  3. Find K Nearest Neighbors │
│  4. Majority Vote for Label  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Output Results              │
│  ├─ Draw Bounding Box        │
│  ├─ Display Label            │
│  ├─ Show Confidence          │
│  └─ Real-time Update         │
└──────────────────────────────┘

Key Algorithms Used:
├─ Thresholding: Binary segmentation
├─ Grass Fire Algorithm: Distance transform for morphology
├─ Region Growing: 4/8-connected component labeling
├─ Moment Invariants: Rotation/scale invariant features
└─ K-NN Classifier: Distance-based classification
```

---

## Features and Capabilities

### Core Capabilities

1. **Training System** - Automated feature extraction and database creation
2. **Real-time Recognition** - Live video object detection and classification
3. **Static Image Analysis** - Detailed analysis of single images with intermediate steps
4. **Multiple Object Detection** - Simultaneous recognition of multiple objects in a scene
5. **Evaluation Tools** - Confusion matrix generation and accuracy metrics
6. **Morphological Processing** - Custom erosion, dilation, and region growing algorithms

### Key Highlights

- **17 Object Classes**: Recognizes diverse categories including wearables, electronics, stationery, and household items
- **83% Accuracy**: Achieves strong classification performance across all classes
- **Real-time Performance**: Processes video at ~30 FPS for single objects
- **Unknown Object Handling**: Dynamically learns new objects through user interaction
- **Custom Algorithms**: Implements morphology and segmentation from scratch

---

## Supported Object Classes

The system is trained to recognize **17 different object types**:

| Category | Objects |
|----------|---------|
| **Wearables** | Beanie, Cap, Glove, Mask, Watch |
| **Electronics** | Phone, Mic, Remote, Ear Buds, Fire Alarm |
| **Stationery** | Book, Pen, Wallet |
| **Household** | Spoon, Umbrella, Glasses Case, Bottle Cap, Chess Board |

!!! info "Unknown Object Handling"
    When an object doesn't match any trained class (distance exceeds threshold), the system prompts the user to provide a new label and automatically adds it to the training database, enabling continuous learning.

---

## Feature Extraction Techniques

### 9-Dimensional Feature Vectors

The system extracts comprehensive feature vectors for each object:

1. **Area Ratio** - Normalized object area relative to image size
2. **Aspect Ratio** - Width-to-height ratio of oriented bounding box
3. **Percent Filled** - Ratio of object pixels to bounding box area
4. **Hu Moments (7 values)** - Rotation, scale, and translation invariant shape descriptors

### Hu Moment Invariants

Hu moments provide powerful shape descriptors that remain constant under geometric transformations:

- **φ₁**: Captures general shape characteristics
- **φ₂**: Measures elongation
- **φ₃**: Captures triangularity
- **φ₄-φ₇**: Higher-order shape characteristics

**Mathematical Foundation:**

```
Normalized Central Moments:
η_pq = μ_pq / (μ₀₀)^((p+q)/2 + 1)

First Hu Moment:
φ₁ = η₂₀ + η₀₂

Second Hu Moment:
φ₂ = (η₂₀ - η₀₂)² + 4η₁₁²
```

!!! tip "Why Hu Moments?"
    These features remain constant under rotation, scaling, and translation, making them ideal for robust object recognition across varying viewpoints and scales.

---

## K-NN Classification Methods

### Algorithm Overview

The K-Nearest Neighbors classifier predicts object labels based on similarity to training samples:

```
1. Extract features from unknown object → query_vector
2. Load all training samples → database
3. For each training sample:
       distance = euclidean_distance(query_vector, sample_vector)
4. Sort by distance (ascending)
5. Select K nearest neighbors
6. Predicted_label = majority_vote(K_neighbor_labels)
```

### Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Euclidean** | `√Σ(a-b)²` | Equal feature importance |
| **Scaled Euclidean** | `√Σw(a-b)²` | Weighted features (prioritize specific features) |

### Choosing K Value

| K Value | Best For | Pros | Cons |
|---------|----------|------|------|
| **K=1** | Small, clean datasets | Fast, simple | Sensitive to noise/outliers |
| **K=3** | General purpose | Balanced, good noise reduction | Slightly slower |
| **K=5** | Large datasets (default) | Robust to outliers | Slower classification |
| **K=7+** | Very noisy datasets | Most stable | May oversmooth, slowest |

!!! note "Tie-breaking Rule"
    If multiple labels have the same vote count (e.g., 2 votes for "Book" and 2 votes for "Phone"), the system selects the label with the **smallest average distance** from the query object.

---

## Benefits and Limitations

!!! success "Benefits"
    - **Fast Training**: Simple feature extraction enables quick model updates
    - **Interpretable**: Clear feature-based classification easy to debug
    - **No GPU Required**: Runs efficiently on standard hardware
    - **Scale/Rotation Invariant**: Hu moments provide robust shape recognition
    - **Real-time Performance**: 30 FPS for single objects, 15 FPS for multiple objects
    - **Continuous Learning**: Unknown object handling enables dynamic expansion

!!! warning "Limitations"
    - **Lighting Sensitivity**: Fixed threshold may fail under varying illumination
    - **Similar Shapes**: Objects with similar geometry can be confused (e.g., Cap vs Bottle Cap)
    - **Occlusion**: Partial visibility significantly reduces accuracy
    - **Background Assumption**: Works best with high-contrast, clean backgrounds
    - **Database Size**: Performance degrades with very large training sets (>1000 samples)
    - **No Color/Texture**: Uses only geometric features, missing discriminative information

---

## Usage Examples

### 1. Training the Classifier

Train the system by providing labeled images:

=== "Windows"
    ```bash
    # Automatic mode (labels from filenames)
    bin\train.exe data\train_images

    # Manual mode (prompt for labels)
    bin\train.exe data\train_images data\db\features.csv m
    ```

=== "Linux/macOS"
    ```bash
    # Automatic mode (labels from filenames)
    ./train data/train_images

    # Manual mode (prompt for labels)
    ./train data/train_images data/db/features.csv m
    ```

**Filename Format for Automatic Mode:**
```
ObjectName###.jpg
Examples: Beanie01.jpg, Cup05.jpg, Phone10.jpg
```

**Output:** Creates `features.csv` with format:
```
filepath,label,area_ratio,aspect_ratio,percent_filled,hu1,hu2,hu3,hu4,hu5,hu6,hu7
```

### 2. Real-time Video Recognition

Recognize objects in live video feed:

=== "Windows"
    ```bash
    bin\Match.exe
    ```

=== "Linux/macOS"
    ```bash
    ./vidDisplay
    ```

**Controls:**

- Press `q` to quit
- Press `s` to save current frame
- System displays recognized label with K-NN confidence

### 3. Multiple Object Detection

Detect and recognize multiple objects in a static image:

=== "Windows"
    ```bash
    bin\MultipleObjects.exe data\example001.png data\db\features.csv euclidean 3
    ```

=== "Linux/macOS"
    ```bash
    ./multipleObjects data/example001.png data/db/features.csv euclidean 3
    ```

**Parameters:**

- `imagePath` - Path to input image
- `featuresFile` - Path to trained features CSV
- `distanceMetric` - `euclidean` or `scaled_euclidean`
- `K` (optional) - Number of nearest neighbors (default: 1)

### 4. Static Image Analysis

Analyze a single image with detailed intermediate steps:

=== "Windows"
    ```bash
    bin\StaticImageAnalyzer.exe data\example068.png
    ```

=== "Linux/macOS"
    ```bash
    ./staticImageAnalyzer data/example068.png
    ```

**Output:** Displays intermediate images:

- Original image
- Binary (thresholded) image
- Cleaned image (after morphology)
- Segmented regions (color-coded)
- Final labeled image

### 5. Confusion Matrix Generation

Evaluate classifier performance:

=== "Windows"
    ```bash
    bin\ConfusionMatrixGenerator.exe data\db\features.csv data\db\confusion_matrix.csv 3
    ```

=== "Linux/macOS"
    ```bash
    ./confusionMatrix data/db/features.csv data/db/confusion_matrix.csv 3
    ```

**How It Works:**

1. Loads all training samples from features CSV
2. For each sample, performs leave-one-out cross-validation:
   - Remove sample from training set
   - Predict label using K-NN on remaining samples
   - Compare prediction with ground truth label
3. Builds confusion matrix: rows = actual labels, columns = predicted labels
4. Calculates per-class and overall accuracy

**Interpreting Results:**

- Diagonal values = correct predictions (higher is better)
- Off-diagonal values = misclassifications (identify confusion patterns)
- Right column shows per-class accuracy
- Bottom row shows overall system accuracy

---

## Performance Metrics

### System Accuracy

The system achieves **83% average accuracy** across 17 object classes using K=5 and scaled Euclidean distance.

**Top Performing Classes:**

| Class | Accuracy |
|-------|----------|
| Glove | 96.4% |
| Beanie | 92.3% |
| Mask | 91.7% |
| Chess Board | 90.0% |
| Ear Buds | 89.3% |
| Remote | 89.5% |
| Spoon | 88.9% |
| Book | 88.2% |

**Challenging Classes:**

| Class | Accuracy | Common Confusions |
|-------|----------|-------------------|
| Cap | 65.0% | Fire Alarm, Wallet |
| Bottle Cap | 69.2% | Cap, Book |
| Fire Alarm | 60.0% | Cap, Book |
| Watch | 70.0% | Mic |

!!! note "Key Observations"
    - Classes with distinctive shapes (Glove, Beanie, Mask) achieve >90% accuracy
    - Similar-shaped objects (Cap vs Bottle Cap) show confusion patterns
    - Training with 10-20 samples per class yields best results

### Real-time Performance

**Training Phase:**

- **Time Complexity**: O(N × M) where N = number of images, M = pixels per image
- **Typical Time**: 100 images (~2-5 minutes)
- **Bottleneck**: Region growing and moment computation

**Recognition Phase:**

- **Time Complexity**: O(K × D) where K = training samples, D = feature dimensions
- **Single object**: ~30 FPS (33ms per frame)
- **Multiple objects (3-5)**: ~15 FPS (66ms per frame)
- **Bottleneck**: K-NN distance computation for large databases

### Optimization Tips

!!! tip "Performance Optimization"
    1. **Reduce Database Size**: Only keep diverse, representative samples
    2. **Limit K Value**: K=1 or K=3 is much faster than K=7+
    3. **Lower Resolution**: Scale down input images (e.g., 640×480 → 320×240)
    4. **Optimize Morphology**: Reduce erosion/dilation iterations if acceptable
    5. **Compile with Optimization**: Use `-O3` flag for 20-30% speedup

---

## Learning Objectives

This project demonstrates key computer vision and pattern recognition concepts:

### Core Algorithms Implemented

1. **Image Segmentation**
   - Binary thresholding (including Otsu's automatic method)
   - Morphological operations (erosion, dilation)
   - Connected component labeling (region growing)

2. **Feature Extraction**
   - Spatial moments computation
   - Central moments (translation invariant)
   - Hu moment invariants (rotation/scale/translation invariant)
   - Geometric features (area ratio, aspect ratio, percent filled)

3. **Classification**
   - K-Nearest Neighbors algorithm
   - Distance metrics (Euclidean, scaled Euclidean)
   - Majority voting with tie-breaking

4. **System Design**
   - Training/testing pipeline separation
   - Real-time video processing
   - Performance evaluation (confusion matrix)

### Advanced Techniques

- **Grass Fire Algorithm**: Custom implementation for distance transforms
- **Stack-based Region Growing**: Efficient connected component analysis
- **Otsu Thresholding**: Automatic threshold selection via variance maximization
- **Unknown Object Handling**: Dynamic database expansion through user interaction

### Practical Skills

- OpenCV integration for image I/O and display
- CSV database management for feature storage
- Real-time video capture and processing
- Performance optimization and profiling
- Debugging computer vision pipelines

---

## Source Code and Documentation

The complete implementation, build instructions, and detailed algorithm explanations are available in the project repository:

**[View Assignment 3 Source Code →](../Assignment3/RT2DOR/)**

For building from source, development setup, and contribution guidelines, see:

**[Development Documentation →](../Assignment3/RT2DOR/DEVELOPMENT.md)**

---

[← Back to Home](index.md)
