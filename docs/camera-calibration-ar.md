# Camera Calibration and Augmented Reality

[← Back to Home](index.md)

## Overview

This project implements a **Camera Calibration and Augmented Reality (AR)** system that calibrates cameras using a chessboard pattern and overlays virtual 3D objects onto real-world scenes in real-time. The system uses computer vision techniques to detect calibration targets, estimate camera pose, and project virtual content that accurately aligns with the physical environment.

The system provides multiple executables for camera calibration, real-time AR projection, feature detection, and advanced AR extensions including ArUco marker tracking and OpenGL 3D model rendering.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CAMERA CALIBRATION & AUGMENTED REALITY                    │
│                            System Workflow Diagram                          │
└─────────────────────────────────────────────────────────────────────────────┘

                   ┌──────────────────────────────────┐
                   │    CALIBRATION PHASE             │
                   │  (Offline - One Time Setup)      │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │ Video Stream (Camera Feed)  │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  For Each Frame:             │
                   │  1. Detect Chessboard        │
                   │     (9×6 internal corners)   │
                   │  2. Extract Corner Points    │
                   │  3. Refine using cornerSubPix│
                   │  4. Display Detected Corners │
                   └──────────────┬───────────────┘
                                  │
                                  │ User presses 's'
                                  ▼
                   ┌──────────────────────────────┐
                   │  Store Calibration Data:     │
                   │  - Image Points (2D corners) │
                   │  - Object Points (3D world)  │
                   │  (Repeat 5+ times)           │
                   └──────────────┬───────────────┘
                                  │
                                  │ User presses 'c'
                                  ▼
                   ┌──────────────────────────────┐
                   │  Camera Calibration:         │
                   │  cv::calibrateCamera()       │
                   │  - Minimize reprojection err │
                   │  - Solve for intrinsics      │
                   │  - Solve for distortion      │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Store to cameraParams.csv   │
                   │  - Camera Matrix (3×3)       │
                   │  - Distortion Coefficients   │
                   │  - Reprojection Error        │
                   └──────────────────────────────┘
                                  │
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                     │
        ▼                                                     ▼
┌────────────────────┐                          ┌────────────────────┐
│ AR PROJECTION PHASE│                          │  EXTENSION PHASE   │
│ (Real-time)        │                          │  (Advanced AR)     │
└────────┬───────────┘                          └────────┬───────────┘
         │                                               │
         ▼                                               ▼
┌────────────────────┐                       ┌──────────────────────┐
│ Load Camera Params │                       │ ArUco Marker AR      │
│ from CSV           │                       │ OpenGL 3D Models     │
└────────┬───────────┘                       │ Static Image/Video   │
         │                                   │ Homography-based     │
         ▼                                   └──────────────────────┘
┌────────────────────────────────┐
│  For Each Frame:               │
│  1. Detect Chessboard          │
│  2. Extract Corners            │
│  3. Estimate Camera Pose       │
│     - solvePnP (rotation)      │
│     - solvePnP (translation)   │
└────────┬───────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Define Virtual Object:      │
│  - 3D World Coordinates      │
│  - Object type:              │
│    - House ('h')             │
│    - Rectangle+Axes ('r')    │
│    - Arrow ('a')             │
│    - Cone ('c')              │
│    - Tetrahedron ('t')       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Project to Image Plane:     │
│  cv::projectPoints()         │
│  - Apply camera matrix       │
│  - Apply distortion coeffs   │
│  - Transform 3D → 2D         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Draw Virtual Object:        │
│  - Connect projected points  │
│  - Draw lines/polygons       │
│  - Add 3D axes (optional)    │
│  - Render with perspective   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Display Results:            │
│  - Original frame            │
│  - Detected chessboard       │
│  - Augmented reality overlay │
└──────────────────────────────┘

Key Algorithms:
├─ Chessboard Detection: cv::findChessboardCorners
├─ Corner Refinement: cv::cornerSubPix
├─ Camera Calibration: cv::calibrateCamera
├─ Pose Estimation: cv::solvePnP
├─ 3D-to-2D Projection: cv::projectPoints
└─ Harris Corner Detection: cv::cornerHarris
```

---

## Key Features

### Core Capabilities

- **Camera Calibration** - Compute intrinsic camera parameters from chessboard images
- **Real-time AR** - Project virtual 3D objects onto detected calibration targets
- **Pose Estimation** - Calculate camera position and orientation relative to target
- **Virtual Object Library** - Multiple pre-defined 3D objects (house, axes, arrow, cone, tetrahedron)
- **Harris Corner Detection** - Feature extraction for advanced computer vision applications
- **Static Image/Video AR** - Project virtual objects onto recorded media

### Camera Calibration Features

The system computes a complete camera model including:

1. **Camera Matrix (Intrinsic Parameters)** - 3×3 matrix encoding:
   - Focal lengths (fx, fy)
   - Principal point (cx, cy)
   - Skew coefficient (usually 0)

2. **Distortion Coefficients** - 5 parameters modeling lens distortion:
   - k1, k2: Radial distortion (barrel/pincushion)
   - p1, p2: Tangential distortion
   - k3: Higher-order radial distortion

3. **Reprojection Error** - Quality metric for calibration accuracy (lower is better)

### Virtual Objects

The system can render the following 3D virtual objects:

- **House (`'h'`)** - 3D house structure with base and roof
- **Rectangle with Axes (`'r'`)** - Outer rectangle boundary with 3D coordinate axes (X, Y, Z)
- **Arrow (`'a'`)** - 3D arrow shape pointing upward
- **Cone (`'c'`)** - 3D cone with circular base
- **Tetrahedron (`'t'`)** - 3D pyramid with triangular base

---

## Technical Details

### Camera Calibration Algorithm

The calibration process uses **Zhang's Method** (1998) to compute camera parameters:

1. **Chessboard Detection** - Detect 9×6 internal corners using `cv::findChessboardCorners`
2. **Corner Refinement** - Sub-pixel accuracy using `cv::cornerSubPix` with 5×5 window
3. **Build Point Correspondences**:
   - **Image Points**: 2D pixel coordinates of detected corners
   - **Object Points**: 3D world coordinates (assuming chessboard at Z=0 plane)
4. **Collect Multiple Views** - Gather 5-15 image-object point pairs from different angles
5. **Optimize Parameters** - Use `cv::calibrateCamera` to minimize reprojection error

**Mathematical Formulation:**
```
Minimize: Σ ||image_points - projectPoints(object_points, rvec, tvec, K, distCoeffs)||²

Where:
  K = Camera matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
  distCoeffs = [k1, k2, p1, p2, k3]
  rvec, tvec = Rotation and translation vectors for each view
```

### Pose Estimation (solvePnP)

Once calibrated, the system estimates camera pose in real-time:

1. **Input**: 3D object points, 2D image points, camera parameters
2. **Output**: Rotation vector (rvec) and translation vector (tvec)
3. **Method**: Iterative Levenberg-Marquardt optimization
4. **Use**: Determines where virtual objects should be placed

**Geometric Interpretation:**
- **rvec**: How much the chessboard is rotated relative to camera (axis-angle representation)
- **tvec**: Where the chessboard is located in camera space (tx, ty, tz)

### 3D-to-2D Projection Pipeline

Virtual objects are projected using `cv::projectPoints`:

```
Step 1: World → Camera Coordinates
    P_camera = R * P_world + tvec

Step 2: Camera → Normalized Image Coordinates
    x_norm = P_camera.x / P_camera.z
    y_norm = P_camera.y / P_camera.z

Step 3: Apply Lens Distortion
    x_distorted = x_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶) + ...
    y_distorted = y_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶) + ...

Step 4: Normalized → Pixel Coordinates
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
```

### Harris Corner Detection

For feature extraction, the system implements Harris corner detection:

```
1. Compute image gradients: Ix, Iy
2. Build structure tensor: M = [Σ(Ix²), Σ(Ix*Iy); Σ(Ix*Iy), Σ(Iy²)]
3. Compute corner response: R = det(M) - k*trace(M)²
4. Threshold and non-maximum suppression
```

---

## Benefits and Limitations

!!! success "Benefits"
    - **High Accuracy**: Reprojection errors < 0.5 pixels achievable with proper calibration
    - **Real-time Performance**: 30+ FPS on modern hardware
    - **Flexible**: Works with any calibrated camera (webcams, smartphones, industrial cameras)
    - **Extensible**: Easy to add new virtual objects and tracking methods
    - **Robust**: Multiple tracking options (chessboard, ArUco markers, homography)
    - **Educational**: Clear demonstration of camera calibration and AR principles

!!! warning "Limitations"
    - **Chessboard Dependency**: Requires visible calibration target for basic AR mode
    - **Lighting Sensitive**: Poor lighting affects chessboard detection accuracy
    - **Single Plane**: Virtual objects anchored to planar target (no SLAM)
    - **No Occlusion Handling**: Virtual objects don't hide behind real objects
    - **Calibration Required**: Each camera needs individual calibration
    - **Limited Interaction**: Virtual objects are static, not interactive

---

## Usage

### 1. Camera Calibration

Calibrate your camera to compute intrinsic parameters:

```bash
# macOS/Linux
./main
```

**Calibration Workflow:**
1. Position 9×6 chessboard pattern in front of camera
2. Wait for detection (green circles indicate successful detection)
3. Press **'s'** to save frame (repeat 5-10 times from different angles)
4. Vary angles by tilting, rotating, and moving the chessboard
5. Press **'c'** to calibrate using all captured frames
6. Check reprojection error (aim for < 1.0 pixels)
7. Press **'q'** to quit and save parameters

**Output:** Creates `resources/cameraParams.csv`:
```csv
cameraMatrix,1303.0435,0.0000,184.4379,0.0000,1383.8424,103.2846,0.0000,0.0000,1.0000
distCoeff,-0.3816,10.6530,-0.0511,0.0079,-87.8130
reprojectionError,0.3992
```

### 2. Real-time AR Projection

Project virtual objects onto the chessboard:

```bash
# macOS/Linux
./project
```

**Controls:**
- Press **'q'** to quit
- Virtual object automatically renders when chessboard detected
- Three windows display: Original, Detected, Augmented

**Customization:** Edit `src/project.cpp`:
- Line 25: Change `virtual_object` ('h', 'r', 'a', 'c', 't')
- Line 29: Update `paramsFile` path if needed

### 3. Harris Corner Detection

Detect and visualize Harris corners in real-time:

```bash
./features
```

**Output:**
- Original camera feed
- Detected corners highlighted with circles

### 4. Static Image/Video AR

Apply AR to static media:

```bash
# Static image AR
./extensions resources/cameraParams.csv i resources/staticImage_1.jpg h

# Static video AR
./extensions resources/cameraParams.csv v resources/staticVid_1.mp4 a
```

**Parameters:**
- `cameraParamsPath` - Path to calibration CSV
- `mode` - `i` (image), `v` (video), `l` (live)
- `staticFilePath` - Path to image/video file
- `virtualObject` - Object type: `h`, `r`, `a`, `c`, `t`

### 5. ArUco Marker AR

Track ArUco markers instead of chessboards:

```bash
./multiTarget
```

**Features:**
- Detects multiple markers simultaneously
- More robust in complex environments
- Uses DICT_6X6_250 dictionary

### 6. OpenGL 3D Model Rendering

Render complex 3D models using OpenGL:

```bash
./openGLExtension
```

**Features:**
- Loads Suzanne 3D model (Blender monkey head)
- Hardware-accelerated rendering (OpenGL 3.3)
- Real-time lighting and shading

---

## Performance Metrics

### Calibration Quality Guidelines

| Reprojection Error | Quality | Action |
|-------------------|---------|--------|
| < 0.5 pixels | Excellent | Ready for AR |
| 0.5 - 1.0 pixels | Good | Acceptable for most uses |
| 1.0 - 2.0 pixels | Acceptable | Consider recalibration |
| > 2.0 pixels | Poor | Recalibrate with more images |

### System Performance

- **Frame Rate**: 30-60 FPS (depends on camera and processing)
- **Calibration Time**: 2-5 minutes (including image capture)
- **Detection Range**: 30cm to 100cm from camera
- **Angle Tolerance**: ±60° from perpendicular view
- **Lighting Requirements**: Even, diffuse lighting without glare

---

## Learning Objectives

This project demonstrates understanding of:

1. **Camera Geometry** - Pinhole camera model, intrinsic/extrinsic parameters
2. **Camera Calibration** - Zhang's method, reprojection error minimization
3. **Lens Distortion** - Radial and tangential distortion models
4. **Pose Estimation** - PnP problem, rotation representations
5. **3D Projections** - World-to-image coordinate transformations
6. **Feature Detection** - Corner detection (chessboard, Harris)
7. **Augmented Reality** - Virtual object overlay with proper perspective
8. **Real-time Processing** - Video stream processing, interactive applications
9. **Computer Vision Pipeline** - Detection → Estimation → Projection → Rendering
10. **Extensions** - ArUco markers, OpenGL rendering, homography methods

---

## Building from Source

For detailed build instructions, dependencies, and troubleshooting, see the [Development Guide](https://github.com/praphul-kumar/CS-5330-Pattern-Recognition-and-Computer-Vision/blob/main/Assignment4/CameraCalibrationAndAR/DEVELOPMENT.md).

**Requirements:**
- OpenCV 4.x with ArUco module
- GLFW, GLEW, GLM, Assimp (for extensions)
- C++11 or later compiler
- CMake or Makefile build system

---

## References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Camera Calibration Method (1998)](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)
- [OpenCV Pose Estimation (solvePnP)](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)

---

[← Back to Home](index.md)
