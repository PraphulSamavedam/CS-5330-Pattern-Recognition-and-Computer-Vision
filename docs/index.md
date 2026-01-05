# CS-5330 Pattern Recognition and Computer Vision

This repository contains projects developed for CS-5330 Pattern Recognition and Computer Vision course. Each assignment explores fundamental computer vision concepts through practical implementations.

**Institution:** Northeastern University    
**Focus Areas:** Deep learning, Image processing, Augmented Reality, object retrieval, feature extraction, object recognition, camera calibration, 

## Technologies

- **OpenCV** - Image processing, camera calibration, AR projection
- **C++** - Core implementations for assignments 1-4
- **PyTorch** - Deep learning framework for assignment 5
- **Python** - Neural network training and testing
- Classical computer vision algorithms
- Deep learning architectures (CNNs, transfer learning)

---

## Projects Overview

### [Real-Time Filtering](real-time-filtering.md)
Real-time image processing with visual effects on live camera feed and static images.

**Key Features:** Negative effect, blur filters, cartoonization, edge detection    
**Concepts:** Color space transformations (RGB, HSV), convolution operations, real-time processing    
**Demo:**    
[![Watch the demo on YouTube](http://img.youtube.com/vi/EXsLRqkdQ3k/mqdefault.jpg)](http://www.youtube.com/watch?v=EXsLRqkdQ3k) 

---

### [Content-Based Image Retrieval (CBIR)](image-retrieval.md)
Sophisticated image retrieval system using histogram-based features and distance metrics.

**Key Features:** 8+ feature extraction techniques, 6 distance metrics, CSV caching (~100x speedup)    
**Concepts:** Histogram-based features, texture analysis (gradient, Laplacian), distance metrics (SSE, histogram intersection)    
**Performance:** Handles 1000+ image databases efficiently    

---

### [Real-Time 2D Object Recognition](object-recognition.md)
Real-time object classification system recognizing 17 object types using K-NN.

**Key Features:** Moment-based features, Hu moments, rotation invariance    
**Concepts:** Morphological operations, region growing, K-NN classification, moment invariants    
**Performance:** 83% average accuracy, 30 FPS single object, 15 FPS multiple objects    

---

### [Camera Calibration & Augmented Reality](camera-calibration-ar.md)
Camera calibration and AR system projecting virtual 3D objects onto real-world scenes.

**Key Features:** Zhang's calibration, pose estimation (solvePnP), 5 virtual objects    
**Concepts:** Camera intrinsics/extrinsics, lens distortion, 3D-to-2D projection, pose estimation    
**Extensions:** ArUco markers, OpenGL 3D models, homography-based projection    

---

### [Character Recognition (Deep Learning)](character-recognition.md)
CNN-based character recognition for handwritten digits and Greek letters with transfer learning.

**Key Features:** LeNet-5 inspired CNN, transfer learning, filter visualization    
**Concepts:** Convolutional Neural Networks, backpropagation, transfer learning, dropout regularization    
**Performance:** 98%+ MNIST accuracy, 90%+ Greek letters with minimal training data    

---

## Repository Structure

```
CS-5330-Pattern-Recognition-and-Computer-Vision/
├── Assignment1/          # Real-Time Filtering
├── Assignment2/          # Content-Based Image Retrieval
├── Assignment3/          # Real-Time 2D Object Recognition
├── Assignment4/          # Camera Calibration and AR
└── Assignment5/          # Character Recognition (Deep Learning)
```

Each assignment directory contains:
- Source code implementations
- Detailed README with approach and results
- Sample images and test data
- Build instructions and usage guides

---

## Getting Started

Each project has its own detailed documentation page (use navigation above). Generally:

1. **Prerequisites**: OpenCV, C++ compiler (for assignments 1-4) or PyTorch (for assignment 5)
2. **Build**: Follow CMake instructions in each assignment's README
3. **Run**: Execute the compiled binary or Python script
4. **Documentation**: Click on project links above for comprehensive guides
---

## Contact

**Author:** Praphul Samavedam    
**GitHub:** [@PraphulSamavedam](https://github.com/PraphulSamavedam)    
