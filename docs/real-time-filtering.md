# Assignment 1: Real-Time Filtering

A real-time image processing application that applies various visual effects to both static images and live camera feed.

## Demo

[![Watch the real time filtering demo](http://img.youtube.com/vi/EXsLRqkdQ3k/mqdefault.jpg)](http://www.youtube.com/watch?v=EXsLRqkdQ3k "Real-Time Filtering Demo")

## Overview

This project demonstrates fundamental image processing techniques through a real-time video filtering application. The system allows users to apply various visual effects to live camera feed or static images, showcasing core computer vision concepts like color space manipulation, convolution operations, and real-time processing.

## Features

### Available Filters

- **Negative Effect** - Inverts image colors
- **Grayscale** - Converts to black and white
- **Blur Filters** - Various blur effects using convolution
- **Cartoonization** - Artistic cartoon-style rendering
- **Edge Detection** - Detects and highlights edges
- **Custom Effects** - User-defined filter combinations

### Capabilities

- Real-time video processing from webcam
- Static image processing from files
- Save processed images (press 's')
- Interactive filter switching
- Smooth performance for real-time applications

## Learning Objectives

Through this assignment, the following concepts were explored:

### Color Space Exploration
- RGB color space fundamentals
- HSV color space for hue/saturation manipulation
- Color channel separation and manipulation
- Colorspace conversion techniques

### Data Types and Applications
- Understanding image representation (8-bit, 16-bit, floating point)
- When to use different data types
- Precision vs. performance tradeoffs
- Type conversions and their implications

### Real-time Video Processing
- Capturing and processing video frames
- Frame rate considerations
- Buffering and memory management
- Performance optimization for real-time constraints

### Filter Applications
- Convolution operations and kernels
- Spatial domain filtering
- Frequency domain concepts
- Filter design and implementation

## Technical Implementation

### Core Technologies
- **OpenCV** for image processing and video capture
- **C++** for performance-critical operations
- Real-time processing pipeline design

### Key Algorithms
- Convolution-based filtering
- Color space transformations
- Edge detection (Sobel, Canny)
- Morphological operations

## Usage

The application provides an interactive interface where users can:

1. Start the application with a video source or image file
2. Switch between different filters using keyboard controls
3. View the effects in real-time
4. Save interesting results for later use

Press 'q' to quit the application at any time.

## Project Structure

```
Assignment1/
├── src/              # Source code
├── include/          # Header files
├── data/             # Sample images
└── README.md         # This file
```

## Demonstration

The demo video above shows the system applying various filters in real-time to live video feed, demonstrating the smooth performance and visual quality of the implemented effects.

---

[← Back to Home](index.md)
