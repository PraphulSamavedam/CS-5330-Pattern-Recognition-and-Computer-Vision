# CS-5330 Pattern Recognition and Computer Vision

This repository contains projects developed for CS-5330 Pattern Recognition and Computer Vision course. Each assignment explores fundamental computer vision concepts through practical implementations.

## Projects

### 1. Assignment 1: Real-Time Filtering

A real-time image processing application that applies various visual effects to both static images and live camera feed.

**Features:**
- Negative effect
- Blur filters
- Cartoonization
- Edge detection
- Save processed images (press 's')

**Learning Objectives:**
- Color space exploration
- Understanding datatypes and their applications
- Real-time video processing
- Applying effects to still images and video streams

**Demo:**

[![Watch the real time filtering demo](http://img.youtube.com/vi/EXsLRqkdQ3k/mqdefault.jpg)](http://www.youtube.com/watch?v=EXsLRqkdQ3k "Real-Time Filtering Demo")

---

### 2. Assignment 2: Content-Based Image Retrieval (CBIR)

A sophisticated image retrieval system that finds similar images from a database based on visual features using histogram-based feature extraction and distance metrics.

**Key Approach:** Combines multiple feature extraction techniques (color histograms, texture analysis, spatial information) with various distance metrics to find visually similar images.

**✅ Benefits:**
- **Fast retrieval**: CSV-based feature caching provides ~100x speedup on subsequent runs
- **Versatile**: 8+ feature extraction techniques for different use cases (color-based, texture-rich, spatially consistent)
- **Lighting robust**: rg chromaticity histogram handles different lighting conditions
- **Interactive**: Both GUI and command-line interfaces for flexible usage
- **Scalable**: Handles large databases (1000+ images) efficiently with caching

**⚠️ Limitations:**
- **No semantic understanding**: Relies on low-level features (color, texture), cannot understand object categories
- **Fixed feature vectors**: Cannot adapt features to specific query types without retraining
- **Requires clean backgrounds**: Performance degrades with cluttered or complex backgrounds
- **Manual feature engineering**: Each feature type designed for specific scenarios, no automatic feature learning
- **Translation/rotation sensitive**: Some features (baseline, center-weighted) require aligned images

**Best For:** Finding images with similar visual appearance (color, texture, spatial layout) in controlled datasets where objects are relatively centered and backgrounds are clean.

**[→ Detailed README](Assignment2/Assignment2/README.md)** | **Feature Extraction Techniques:** Baseline (9×9), 2D/3D Histograms, Multi-Histogram, Texture & Color, Quarters, Center & Texture | **Distance Metrics:** SSE, MSE, Histogram Intersection, Entropy, Weighted Histogram

**Learning Objectives:**
- Understanding image feature extraction and representation
- Color space analysis (RGB, rg chromaticity, HSV)
- Texture analysis using gradient magnitude and Laplacian
- Distance metric selection and implementation
- Performance optimization with feature caching
- Building interactive computer vision applications

---

### 3. Assignment 3: Real-Time 2D Object Recognition (RT2DOR)

A real-time object recognition system that classifies 17 object types using moment-based features and K-Nearest Neighbors, achieving 83% average accuracy.

**Key Approach:** Combines traditional computer vision (thresholding, morphology, region growing) with geometric features (Hu moments, aspect ratio, percent filled) and K-NN classification for real-time object recognition.

**✅ Benefits:**
- **Real-time performance**: 30 FPS for single objects, 15 FPS for multiple objects
- **Rotation invariant**: Hu moments provide scale and rotation invariance
- **Explainable features**: Geometric features (area, aspect ratio) are interpretable
- **Custom implementations**: Built-from-scratch morphological operations and segmentation algorithms
- **High accuracy on distinct shapes**: 90%+ accuracy for objects with distinctive shapes (glove 96%, beanie 92%)
- **Unknown object handling**: Automatically prompts for new labels when unknown objects detected

**⚠️ Limitations:**
- **Requires clean backgrounds**: Thresholding assumes objects darker than background
- **Struggles with similar shapes**: 60-70% accuracy for similar objects (cap vs bottle cap vs fire alarm)
- **Fixed threshold**: Manual threshold (124) doesn't adapt to varying lighting
- **Limited texture discrimination**: Primarily geometric features, minimal texture analysis
- **Single viewpoint**: 2D features cannot handle 3D object rotation or varying perspectives
- **Training data needed**: Requires 10-20 samples per object class for good accuracy

**Best For:** Real-time recognition of objects with distinctive geometric shapes on clean backgrounds with consistent lighting (e.g., lab settings, controlled environments).

**[→ Detailed README](Assignment3/RT2DOR/README.md)** | **17 Object Classes** | **83% Average Accuracy** | **Features:** Area Ratio, Aspect Ratio, Percent Filled, 7 Hu Moments | **Algorithms:** Grassfire Transform, Region Growing, Union-Find, K-NN, Otsu's Thresholding

**Learning Objectives:**
- Foreground & background image analysis
- Morphological operators (erosion, dilation)
- Segmentation techniques (region growing, connected components)
- Moments & Hu moments for rotation-invariant object characterization
- K-Nearest Neighbors classification
- Label training and prediction strategies
- Building from-scratch computer vision algorithms

---

### 4. Assignment 4: Camera Calibration and Augmented Reality

A camera calibration and augmented reality system that projects virtual 3D objects onto real-world scenes using chessboard-based pose estimation.

**Key Approach:** Uses Zhang's calibration method with chessboard patterns to compute camera intrinsics, then applies solvePnP for pose estimation to project virtual 3D objects with accurate perspective and distortion correction.

**✅ Benefits:**
- **High calibration accuracy**: Achieves <0.5 pixel reprojection error with proper calibration
- **Real-time AR**: Smooth projection of virtual objects onto detected targets
- **Distortion correction**: Handles radial and tangential lens distortion for wide-angle cameras
- **Multiple virtual objects**: 5 pre-defined 3D objects (house, axes, arrow, cone, tetrahedron)
- **Extensible**: ArUco marker tracking, OpenGL 3D models, static image/video AR, homography-based projection
- **Educational**: Clear visualization of camera parameters and projection pipeline

**⚠️ Limitations:**
- **Requires calibration target**: Chessboard must be visible for pose estimation
- **Single plane tracking**: Standard approach limited to planar surfaces (chessboard)
- **Extreme angle degradation**: Accuracy drops when chessboard viewed at >60° angles
- **No occlusion handling**: Virtual objects don't hide behind real objects
- **Fixed virtual content**: Pre-defined objects, no dynamic or interactive content
- **Lighting independent**: Virtual objects don't match real-world lighting conditions

**Best For:** Educational demonstrations of camera geometry, AR prototyping with known targets, and understanding 3D-to-2D projection mathematics in controlled environments.

**[→ Detailed README](Assignment4/CameraCalibrationAndAR/README.md)** | **Features:** Camera calibration, Real-time AR, Pose estimation (solvePnP), Virtual objects library | **Extensions:** ArUco markers, OpenGL 3D models, Static image/video AR, Harris corner detection

**Learning Objectives:**
- Camera calibration techniques (Zhang's method)
- Intrinsic camera parameters (focal length, principal point)
- Lens distortion modeling (radial and tangential)
- Pose estimation (rotation and translation vectors)
- 3D-to-2D projection mathematics
- Augmented reality fundamentals
- Corner detection algorithms (Harris corners)
- Real-time video processing with geometric transformations

---

### 5. Assignment 5: Character Recognition using Deep Neural Networks

A deep learning system that recognizes handwritten digits (MNIST) and Greek letters using Convolutional Neural Networks (CNNs) with transfer learning, achieving 98%+ accuracy on MNIST.

**Key Approach:** Implements LeNet-5 inspired CNN architecture with PyTorch, then leverages transfer learning to adapt the pre-trained model for Greek letter recognition with minimal training data.

**✅ Benefits:**
- **High accuracy**: 98%+ on MNIST with only 5 epochs of training
- **Transfer learning efficiency**: Achieves 90%+ accuracy on Greek letters with just 27 images per class
- **Fast training**: Transfer learning requires only 45 seconds vs. hours from scratch
- **Feature reusability**: Pre-learned edge/curve detectors work across different character sets
- **Minimal data requirements**: 99% fewer trainable parameters when freezing convolutional layers
- **Explainable filters**: Visualization shows learned features (edge detectors, corner detectors)

**⚠️ Limitations:**
- **Black box nature**: Deep learning models difficult to interpret beyond filter visualization
- **Requires GPU for large-scale**: CPU training acceptable for MNIST but slow for larger datasets
- **Overfitting on small datasets**: Transfer learning helps but still needs careful regularization (dropout)
- **Fixed architecture**: No automatic architecture search, manual design required
- **Domain similarity required**: Transfer learning assumes source (digits) and target (Greek letters) share low-level features
- **Memory intensive**: Batch processing and model storage require significant RAM

**Best For:** Character recognition tasks with limited training data where similar tasks have abundant data for pre-training. Ideal for rapid prototyping and educational understanding of CNNs and transfer learning.

**[→ Detailed README](Assignment5/CharacterRecognition/README.md)** | **MNIST Accuracy:** 98%+ | **Greek Letters:** 90%+ with transfer learning | **Architecture:** LeNet-5 inspired (2 conv layers, 2 FC layers, dropout) | **Framework:** PyTorch

**Learning Objectives:**
- Convolutional Neural Networks (CNN) architecture design
- Deep learning training pipeline (forward pass, loss computation, backpropagation)
- Transfer learning and feature reusability
- Regularization techniques (dropout, weight decay)
- Hyperparameter tuning (learning rate, momentum, batch size)
- PyTorch framework fundamentals
- Filter visualization and understanding learned features
- Data preprocessing and normalization for neural networks

---

## Technologies

- **OpenCV** - Image processing, camera calibration, AR projection
- **C++** - Core implementations for assignments 1-4
- **PyTorch** - Deep learning framework for assignment 5
- **Python** - Neural network training and testing
- Classical computer vision algorithms
- Deep learning architectures (CNNs, transfer learning)

## Course Information

**Course:** CS-5330 Pattern Recognition and Computer Vision
**Focus Areas:** Image processing, feature extraction, object recognition, camera calibration, augmented reality 
