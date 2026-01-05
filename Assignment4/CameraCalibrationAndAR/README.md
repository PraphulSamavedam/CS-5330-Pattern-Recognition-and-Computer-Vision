# Assignment 4: Camera Calibration and Augmented Reality

## Overview

This project implements a **Camera Calibration and Augmented Reality (AR)** system that calibrates cameras using a chessboard pattern and overlays virtual 3D objects onto real-world scenes in real-time. The system uses computer vision techniques to detect calibration targets, estimate camera pose, and project virtual content that accurately aligns with the physical environment.

The system provides multiple executables for camera calibration, real-time AR projection, feature detection, and advanced AR extensions including ArUco marker tracking and OpenGL 3D model rendering.

## Camera Calibration and AR System Architecture

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

Key Algorithms Used:
├─ Chessboard Detection: cv::findChessboardCorners
├─ Corner Refinement: cv::cornerSubPix
├─ Camera Calibration: cv::calibrateCamera
├─ Pose Estimation: cv::solvePnP
├─ 3D-to-2D Projection: cv::projectPoints
└─ Harris Corner Detection: cv::cornerHarris (features executable)
```

## Features

### Core Capabilities

1. **Camera Calibration** - Compute intrinsic camera parameters from chessboard images
2. **Real-time AR** - Project virtual 3D objects onto detected calibration targets
3. **Pose Estimation** - Calculate camera position and orientation relative to target
4. **Virtual Object Library** - Multiple pre-defined 3D objects (house, axes, arrow, cone, tetrahedron)
5. **Harris Corner Detection** - Feature extraction for advanced computer vision applications
6. **Static Image/Video AR** - Project virtual objects onto recorded media

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

1. **House (`'h'`)** - 3D house structure with base and roof
2. **Rectangle with Axes (`'r'`)** - Outer rectangle boundary with 3D coordinate axes (X, Y, Z)
3. **Arrow (`'a'`)** - 3D arrow shape pointing upward
4. **Cone (`'c'`)** - 3D cone with circular base
5. **Tetrahedron (`'t'`)** - 3D pyramid with triangular base

### Extensions Implemented

1. **ArUco Marker Tracking** - Detect and track ArUco markers for robust AR
2. **OpenGL 3D Model Rendering** - Render complex 3D models (Suzanne from Blender) using OpenGL
3. **Static Image AR** - Apply AR to still images
4. **Static Video AR** - Apply AR to recorded video files
5. **Homography-based Projection** - Alternative projection method for planar surfaces
6. **Multi-Target Detection** - Track multiple chessboards or markers simultaneously

## Quick Start

### Requirements
- **OpenCV 4.x** with ArUco module - For image processing, camera calibration, and marker detection
- **GLFW** - For OpenGL window management (extensions only)
- **GLEW** - OpenGL Extension Wrangler Library (extensions only)
- **GLM** - OpenGL Mathematics library (extensions only)
- **Assimp** - Asset Import Library for 3D models (extensions only)
- Pre-compiled executables available (macOS)
- For building from source, see [DEVELOPMENT.md](DEVELOPMENT.md)

### Using Pre-compiled Executables (macOS)

The system comes with pre-compiled executables:
- `main` - Camera calibration program
- `project` - Real-time AR projection
- `features` - Harris corner detection

**Note:** OpenCV libraries must be available in your system library path.

### Building from Source

For building from source (Linux/macOS/Windows), see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Usage

### 1. Camera Calibration

First, calibrate your camera to compute intrinsic parameters:

**macOS/Linux:**
```bash
./main
```

**Calibration Workflow:**
1. **Position the chessboard** - Hold a printed 9×6 chessboard pattern in front of the camera
2. **Wait for detection** - Green circles appear when corners are detected and refined
3. **Press 's' to save** - Capture current frame's corner points (repeat 5-10 times from different angles)
4. **Vary the angles** - Tilt, rotate, and move the chessboard to different positions for better calibration
5. **Press 'c' to calibrate** - Compute camera parameters using all captured frames
6. **Check quality** - System displays reprojection error (aim for < 1.0 pixels)
7. **Press 'q' to quit** - Exit and save parameters to `resources/cameraParams.csv`

**Output:** Creates `resources/cameraParams.csv` with format:
```csv
cameraMatrix,fx,0,cx,0,fy,cy,0,0,1
distCoeff,k1,k2,p1,p2,k3
reprojectionError,error_value
```

**Example output:**
```csv
cameraMatrix,1303.0435,0.0000,184.4379,0.0000,1383.8424,103.2846,0.0000,0.0000,1.0000
distCoeff,-0.3816,10.6530,-0.0511,0.0079,-87.8130
reprojectionError,0.3992
```

**Calibration Quality Guidelines:**
- **Excellent**: Reprojection error < 0.5 pixels
- **Good**: 0.5 - 1.0 pixels
- **Acceptable**: 1.0 - 2.0 pixels
- **Poor**: > 2.0 pixels (recalibrate with more images)

**Tips for Better Calibration:**
- Use a rigid, flat chessboard (print on cardboard or foam board)
- Ensure consistent lighting without glare or shadows
- Capture 10-15 images from various angles and distances
- Include images with chessboard near corners and center of frame
- Avoid motion blur - hold steady when pressing 's'

### 2. Real-time AR Projection

After calibration, project virtual objects onto the chessboard:

**macOS/Linux:**
```bash
./project
```

**Controls:**
- Press `q` to quit
- Virtual object is automatically rendered when chessboard is detected
- Three windows display:
  - **Original** - Raw camera feed
  - **Detected** - Chessboard with detected corners
  - **Augmented** - Final AR result with virtual object overlay

**Configuration:** Edit `src/project.cpp` to customize:
- `virtual_object` (line 25) - Change object type: `'h'` (house), `'r'` (rectangle+axes), `'a'` (arrow), `'c'` (cone), `'t'` (tetrahedron)
- `paramsFile` (line 29) - Path to camera parameters CSV

### 3. Harris Corner Detection

Detect and visualize Harris corners in real-time:

**macOS/Linux:**
```bash
./features
```

**Output:**
- **Image** - Original camera feed
- **corners Image** - Detected Harris corners highlighted with circles

**Use Case:** Feature extraction for advanced computer vision tasks like feature matching, tracking, or SLAM.

### 4. Advanced AR Extensions

#### Static Image AR

Project virtual objects onto a static image:

**Usage:**
```bash
# Command format
./extensions <cameraParamsPath> <mode> <staticFilePath> <virtualObject>

# Example
./extensions resources/cameraParams.csv i resources/staticImage_1.jpg h
```

**Parameters:**
- `cameraParamsPath` - Path to camera calibration CSV (default: `resources/cameraParams.csv`)
- `mode` - `i` for static image, `v` for static video, `l` for live (default: `l`)
- `staticFilePath` - Path to image/video file (required for `i` or `v` modes)
- `virtualObject` - Object type: `h`, `r`, `a`, `c`, `t` (default: `h`)

#### Static Video AR

Apply AR to a recorded video:

```bash
./extensions resources/cameraParams.csv v resources/staticVid_1.mp4 a
```

**Output:** Plays the video with virtual objects overlaid on detected chessboards in each frame.

#### ArUco Marker AR

Track ArUco markers instead of chessboards:

```bash
./multiTarget
```

**Features:**
- Detects multiple ArUco markers simultaneously
- More robust than chessboard detection in complex environments
- Supports DICT_6X6_250 dictionary

#### OpenGL 3D Model AR

Render complex 3D models using OpenGL:

```bash
./openGLExtension
```

**Features:**
- Loads and renders Suzanne 3D model (Blender monkey head)
- Hardware-accelerated rendering using OpenGL 3.3 core profile
- Real-time lighting and shading
- Requires GLFW, GLEW, GLM, and Assimp libraries

## How It Works

### System Pipeline

#### Phase 1: Camera Calibration (Offline)

1. **Chessboard Detection** - Detect 9×6 internal corners using `cv::findChessboardCorners`
2. **Corner Refinement** - Sub-pixel accuracy using `cv::cornerSubPix` with 5×5 window
3. **Build Point Correspondences**:
   - **Image Points**: 2D pixel coordinates of detected corners
   - **Object Points**: 3D world coordinates (assuming chessboard at Z=0 plane)
4. **Collect Multiple Views** - Gather 5-15 image-object point pairs from different angles
5. **Camera Calibration** - Use `cv::calibrateCamera` to minimize reprojection error:
   - Optimize camera matrix (fx, fy, cx, cy)
   - Optimize distortion coefficients (k1, k2, p1, p2, k3)
   - Compute rotation and translation vectors for each view
6. **Store Parameters** - Save camera matrix, distortion coefficients, and error to CSV

#### Phase 2: AR Projection (Real-time)

1. **Load Calibration** - Read camera parameters from `resources/cameraParams.csv`
2. **Video Capture** - Open camera and start video stream
3. **For Each Frame**:
   - **Detect Target** - Find chessboard corners in current frame
   - **Pose Estimation** - Use `cv::solvePnP` to compute camera pose:
     - Rotation vector (rvec): 3D orientation
     - Translation vector (tvec): 3D position
   - **Define Virtual Object** - Create 3D world coordinates for selected object
   - **Project to Image** - Use `cv::projectPoints` to transform 3D → 2D:
     - Apply camera matrix
     - Apply distortion coefficients
     - Apply rotation and translation
   - **Draw Object** - Connect projected 2D points with lines to form object
   - **Display Result** - Show augmented frame with virtual object overlay

## Key Algorithms Explained

### 1. Chessboard Corner Detection

**Purpose:** Automatically locate calibration pattern corners in an image.

**Algorithm (`cv::findChessboardCorners`):**
```
Input: Grayscale image, pattern size (9×6)
1. Apply adaptive thresholding to create binary image
2. Find connected components (potential squares)
3. Filter based on shape, size, and aspect ratio
4. Group squares into rows and columns
5. Verify grid structure (9 columns × 6 rows)
6. Order corners consistently (top-left to bottom-right)
Output: 54 corner points (9×6) in image coordinates
```

**Refinement (`cv::cornerSubPix`):**
```
Input: Grayscale image, initial corner locations
1. For each corner:
   - Define 5×5 pixel window around initial estimate
   - Compute image gradients within window
   - Find point where gradients in perpendicular directions balance
   - Update corner position to sub-pixel accuracy
2. Iterate until convergence (max 100 iterations or 0.001 pixel change)
Output: Refined corners with sub-pixel precision
```

**Use Case:** Provides accurate 2D-3D correspondences for camera calibration.

---

### 2. Building Point Sets

**Purpose:** Create 3D world coordinates corresponding to detected 2D image corners.

**Algorithm (`buildPointsSet`):**
```
Input: corners (54 points), pointsPerRow=9, pointsPerColumn=6
Output: 3D object points in world coordinate system

Assumption: Chessboard lies flat on Z=0 plane

For each corner (index 0 to 53):
    x = index % pointsPerRow      // Column: 0, 1, ..., 8
    y = -(index / pointsPerRow)   // Row: 0, -1, ..., -5 (negative Y-axis)
    z = 0                         // Flat plane
    points.push_back(Vec3f(x, y, z))

Example corner positions:
Corner 0  → (0, 0, 0)    [Top-left]
Corner 8  → (8, 0, 0)    [Top-right]
Corner 45 → (0, -5, 0)   [Bottom-left]
Corner 53 → (8, -5, 0)   [Bottom-right]
```

**Coordinate System:**
- **X-axis**: Increases left to right (0 to 8)
- **Y-axis**: Increases top to bottom (0 to -5, negative convention)
- **Z-axis**: Perpendicular to chessboard plane (always 0 for calibration target)

**Use Case:** Establishes known 3D geometry for calibration and pose estimation.

---

### 3. Camera Calibration

**Purpose:** Compute intrinsic camera parameters and lens distortion model.

**Algorithm (`cv::calibrateCamera`):**
```
Input:
- image_points_list: [N images × 54 corners × 2D coordinates]
- object_points_list: [N images × 54 corners × 3D coordinates]
- image_size: (width, height)

Objective: Minimize reprojection error
    error = Σ ||image_points - projectPoints(object_points, rvec, tvec, K, distCoeffs)||²

Optimization variables:
- K: Camera matrix (3×3)
    [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
  Where:
    fx, fy = focal lengths (pixels)
    cx, cy = principal point (image center, pixels)

- distCoeffs: Distortion coefficients [k1, k2, p1, p2, k3]
    Radial distortion: k1, k2, k3
    Tangential distortion: p1, p2

- rvec[i], tvec[i]: Pose for each calibration image i

Distortion Model:
    x_distorted = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
    y_distorted = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
    where r² = x² + y²

Algorithm Steps:
1. Initialize K with reasonable defaults (focal length ≈ image width)
2. Use Levenberg-Marquardt optimization to minimize reprojection error
3. Iterate until convergence or max iterations (30)
4. Compute final reprojection error

Output:
- Camera matrix K
- Distortion coefficients
- Rotation and translation vectors for each image
- RMS reprojection error
```

**Reprojection Error:**
```
For each calibration image:
    1. Project 3D object points back to 2D using estimated parameters
    2. Compute Euclidean distance between projected and detected corners
    3. RMS error = sqrt(mean(distances²))

Interpretation:
- < 0.5 pixels: Excellent calibration
- 0.5 - 1.0: Good calibration
- 1.0 - 2.0: Acceptable
- > 2.0: Poor, recalibrate
```

**Use Case:** Essential for accurate AR - without proper calibration, virtual objects won't align correctly with the real world.

---

### 4. Pose Estimation (solvePnP)

**Purpose:** Determine 3D position and orientation of camera relative to target.

**Algorithm (`cv::solvePnP`):**
```
Input:
- object_points: 3D world coordinates (54 chessboard corners)
- image_points: 2D pixel coordinates (detected corners)
- camera_matrix: Intrinsic parameters from calibration
- dist_coeffs: Lens distortion model

Objective: Find rotation R and translation t such that:
    image_points ≈ project(R * object_points + t, camera_matrix, dist_coeffs)

Perspective-n-Point (PnP) Problem:
    Given n 3D-2D correspondences, find camera pose

Methods:
1. ITERATIVE (default): Iterative Levenberg-Marquardt refinement
2. P3P: Closed-form solution for exactly 3 points
3. EPNP: Efficient PnP for n ≥ 4 points
4. AP3P: Algebraic P3P variant

Output:
- rvec: Rotation vector (3×1)
  - Axis-angle representation: direction = axis, magnitude = angle (radians)
  - Convert to rotation matrix R using cv::Rodrigues()
- tvec: Translation vector (3×1)
  - (tx, ty, tz) = 3D position of chessboard origin in camera coordinates

Geometric Interpretation:
- R: How much the chessboard is rotated relative to camera
- t: Where the chessboard is located in camera space
```

**Example Pose:**
```
rvec = [0.1, 0.2, 0.05]  → Small rotation around X and Y axes
tvec = [5, -3, 50]       → Object 5 units right, 3 down, 50 units away
```

**Use Case:** Enables placing virtual objects at correct 3D position relative to detected target.

---

### 5. Virtual Object Definition

**Purpose:** Define 3D geometry of virtual objects in world coordinates.

**Algorithm (`buildVirtualObjectPoints`):**

#### House Object (`'h'`)
```
Base Floor (4 corners):
    (0, 0, 0), (4, 0, 0), (4, -4, 0), (0, -4, 0)

Roof (4 corners at height=4):
    (0, 0, 4), (4, 0, 4), (4, -4, 4), (0, -4, 4)

Roof Peak:
    (2, -2, 6)  [Center top]

Total: 9 points defining house edges and roof peak
```

#### Rectangle with Axes (`'r'`)
```
Outer Rectangle (4 corners):
    (-1, 1, 0), (-1, -6, 0), (9, -6, 0), (9, 1, 0)

3D Axes (origin + 3 directions):
    Origin: (0, 0, 0)
    X-axis: (1, 0, 0)  [Red]
    Y-axis: (0, 1, 0)  [Green]
    Z-axis: (0, 0, 1)  [Blue]

Axis Labels (positions for text):
    X: (1.5, 0, 0)
    Y: (0, 1.5, 0)
    Z: (0, 0, 1.5)

Total: 11 points (rectangle + axes + labels)
```

#### Arrow (`'a'`)
```
Shaft (4 corners, vertical):
    Base: (3, -2, 0), (5, -2, 0), (5, -4, 0), (3, -4, 0)
    Top: (3, -2, 4), (5, -2, 4), (5, -4, 4), (3, -4, 4)

Arrowhead (4 points):
    Base: (2, -1, 4), (6, -1, 4), (6, -5, 4), (2, -5, 4)
    Tip: (4, -3, 6)

Total: 13 points
```

**Coordinate System Convention:**
- **X**: Horizontal (left-right on chessboard)
- **Y**: Horizontal (top-bottom on chessboard, negative = down)
- **Z**: Vertical (perpendicular to chessboard, positive = up/away from board)

**Use Case:** Defines 3D structure that will be projected onto 2D image plane.

---

### 6. 3D-to-2D Projection

**Purpose:** Transform 3D world coordinates to 2D pixel coordinates using camera model.

**Algorithm (`cv::projectPoints`):**
```
Input:
- object_points: 3D virtual object coordinates
- rvec, tvec: Camera pose from solvePnP
- camera_matrix: Intrinsic parameters
- dist_coeffs: Lens distortion

Projection Pipeline:

Step 1: World → Camera Coordinates
    Convert rotation vector to matrix: R = Rodrigues(rvec)
    For each 3D point P_world:
        P_camera = R * P_world + tvec

Step 2: Camera → Normalized Image Coordinates
    x_norm = P_camera.x / P_camera.z
    y_norm = P_camera.y / P_camera.z

Step 3: Apply Lens Distortion
    r² = x_norm² + y_norm²
    x_distorted = x_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶) +
                  2*p1*x_norm*y_norm + p2*(r² + 2*x_norm²)
    y_distorted = y_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶) +
                  p1*(r² + 2*y_norm²) + 2*p2*x_norm*y_norm

Step 4: Normalized → Pixel Coordinates
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    Where camera_matrix K = [fx  0  cx]
                             [0  fy  cy]
                             [0   0   1]

Output: 2D pixel coordinates (u, v) for each 3D point
```

**Example Projection:**
```
3D Point: (2, -3, 50)  [2 units right, 3 down, 50 units away]

After rotation/translation:
P_camera = (1.8, -2.9, 48.5)

Normalized:
x_norm = 1.8/48.5 = 0.037
y_norm = -2.9/48.5 = -0.060

After distortion (minimal for well-calibrated cameras):
x_distorted ≈ 0.037
y_distorted ≈ -0.060

Pixel coordinates (assuming fx=1300, fy=1380, cx=320, cy=240):
u = 1300 * 0.037 + 320 = 368 pixels
v = 1380 * (-0.060) + 240 = 157 pixels

Result: Point projects to pixel (368, 157) in image
```

**Use Case:** Enables rendering virtual objects with correct perspective, occlusion, and distortion.

---

### 7. Drawing Virtual Objects

**Purpose:** Render 3D objects onto 2D image using projected points.

**Algorithm (`drawVirtualObject`):**
```
Input:
- image: Frame to draw on
- vir_obj_img_pts: 2D projected pixel coordinates
- object: Object type ('h', 'r', 'a', etc.)

Drawing Strategy:

For House ('h'):
    1. Draw base (4 lines forming square)
       line(pt[0], pt[1]), line(pt[1], pt[2]), line(pt[2], pt[3]), line(pt[3], pt[0])
    2. Draw vertical edges (4 lines from base to roof)
       line(base[i], roof[i]) for i = 0 to 3
    3. Draw roof edges (4 lines)
    4. Draw roof peak (4 lines from roof corners to peak)

For Rectangle + Axes ('r'):
    1. Draw rectangle (4 lines)
    2. Draw 3D axes:
       X-axis: line(origin, x_point) in RED
       Y-axis: line(origin, y_point) in GREEN
       Z-axis: line(origin, z_point) in BLUE
    3. Add text labels at axis endpoints

For Arrow ('a'):
    1. Draw shaft (8 lines forming rectangular prism)
    2. Draw arrowhead base (4 lines)
    3. Draw arrowhead tip (4 lines from base to tip)

Visual Properties:
- Line thickness: 2-3 pixels
- Anti-aliasing: LINE_AA for smooth edges
- Colors: Typically green (0, 255, 0), or RGB for axes
- Filled polygons: Use cv::fillPoly for solid faces (optional)

Depth Handling:
- No explicit Z-buffering (OpenCV doesn't support)
- Draw back faces first, then front faces (painter's algorithm)
- Or compute face normals and skip back-facing polygons
```

**Example Drawing Sequence for House:**
```
1. Base: Connect points [0,1,2,3] with closed loop
2. Walls: Draw lines from base[i] to roof[i]
3. Roof: Connect points [4,5,6,7] with closed loop
4. Peak: Draw lines from roof corners to peak point[8]
```

**Use Case:** Final step in AR pipeline - makes virtual object visible with realistic perspective.

---

### 8. Harris Corner Detection

**Purpose:** Detect interest points (corners) in images for feature matching and tracking.

**Algorithm (`cv::cornerHarris`):**
```
Input: Grayscale image I
Output: Corner response map (higher values = stronger corners)

Step 1: Compute Image Gradients
    Ix = ∂I/∂x  (horizontal gradient, Sobel filter)
    Iy = ∂I/∂y  (vertical gradient, Sobel filter)

Step 2: Compute Structure Tensor (M-matrix) for each pixel
    For each pixel, compute in local window W (e.g., 5×5):
        A = Σ(Ix²)    [Sum of squared x-gradients]
        B = Σ(Iy²)    [Sum of squared y-gradients]
        C = Σ(Ix*Iy)  [Sum of gradient products]

    M = [A  C]
        [C  B]

Step 3: Compute Corner Response
    R = det(M) - k * trace(M)²
    Where:
        det(M) = A*B - C²
        trace(M) = A + B
        k = 0.04 (sensitivity parameter)

    Interpretation:
        R > threshold  → Corner
        R < 0          → Edge
        R ≈ 0          → Flat region

Step 4: Non-Maximum Suppression
    For each corner candidate:
        Check if R is local maximum in neighborhood
        Suppress if neighbors have higher response

Step 5: Threshold and Normalize
    threshold = 0.01 * max(R)
    Keep only corners where R > threshold
    Normalize R to [0, 255] for visualization

Output: Binary image with corners marked
```

**Mathematical Intuition:**
```
Structure Tensor Eigenvalues (λ1, λ2) indicate:
- Both large → Corner (gradients in multiple directions)
- One large  → Edge (gradient in one direction)
- Both small → Flat (no gradients)

Harris uses R = det(M) - k*trace(M)² as computationally cheaper alternative:
    R ≈ λ1*λ2 - k*(λ1 + λ2)²
```

**Parameters:**
- `blockSize = 2`: Neighborhood size for gradient computation
- `apertureSize = 3`: Sobel filter size
- `k = 0.04`: Harris detector free parameter (0.04-0.06 typical)

**Use Case:** Feature detection for tracking, image stitching, SLAM, or object recognition.

---

## Algorithm Workflow Example

### Complete Pipeline for Real-time AR

**Input:** Live video stream, calibrated camera, chessboard target

**Step 1: Calibration (One-time)**
```
1. Capture 10 chessboard images from different angles
2. Detect 54 corners per image → 540 2D points
3. Generate 54 object points per image → 540 3D points
4. Run cv::calibrateCamera:
   Optimize: fx=1303, fy=1384, cx=184, cy=103
            k1=-0.38, k2=10.65, p1=-0.05, p2=0.008, k3=-87.81
   Reprojection error: 0.40 pixels (Excellent!)
5. Save to cameraParams.csv
```

**Step 2: AR Frame Loop (Real-time)**
```
Frame 1:
├─ Capture frame (640×480 RGB)
├─ Detect chessboard: SUCCESS (54 corners found)
├─ Estimate pose:
│  rvec = [0.12, 0.25, 0.03] → Slight tilt
│  tvec = [8, -5, 55]        → 55cm away, slightly right and down
├─ Define house object (9 3D points)
├─ Project 3D → 2D:
│  3D point (0,0,0) → 2D pixel (320, 240) [Base corner]
│  3D point (2,-2,6) → 2D pixel (340, 190) [Roof peak]
│  ...
├─ Draw house:
│  Base square, 4 walls, roof, peak
└─ Display augmented frame

Frame 2:
├─ Camera moved slightly
├─ New pose: tvec = [6, -4, 52] (closer)
├─ Virtual house updates position accordingly
└─ Maintains alignment with chessboard

Frame 3:
├─ Chessboard lost (moved out of view)
├─ Skip AR rendering
└─ Show original frame only
```

**Step 3: Example Projection Calculation**
```
House base corner: (0, 0, 0) in world coordinates

Camera pose from solvePnP:
    R = [[0.98, -0.01, 0.15],     [Slight rotation]
         [0.02,  0.99, 0.05],
         [-0.15, -0.06, 0.98]]
    t = [8, -5, 55]              [8cm right, 5cm down, 55cm away]

Transform to camera coordinates:
    P_camera = R * (0,0,0) + t = t = [8, -5, 55]

Normalize:
    x_norm = 8/55 = 0.145
    y_norm = -5/55 = -0.091

Apply distortion (minimal):
    x_dist ≈ 0.145
    y_dist ≈ -0.091

Project to pixels (K = [1303,0,184; 0,1384,103; 0,0,1]):
    u = 1303 * 0.145 + 184 = 373 pixels
    v = 1384 * (-0.091) + 103 = -23 pixels (off-screen, clipped)

Result: Base corner projects near (373, -23), slightly above visible frame
```

---

## Choosing Algorithms and Parameters

### Calibration Quality Factors

| **Factor** | **Impact** | **Recommendation** |
|------------|------------|--------------------|
| Number of calibration images | More images = better accuracy | 10-15 images minimum |
| Angle variation | Wide range = better generalization | Cover ±45° tilt, rotation |
| Distance variation | Multiple depths = robust focal length | Near (30cm) to far (100cm) |
| Chessboard quality | Flat, rigid = accurate corners | Mounted on rigid board |
| Lighting | Even, no glare = reliable detection | Diffuse lighting, avoid shadows |

### Virtual Object Selection

| **Object** | **Best For** | **Pros** | **Cons** |
|------------|--------------|----------|----------|
| House (`'h'`) | Architectural visualization | Recognizable structure | Complex geometry |
| Rectangle + Axes (`'r'`) | Coordinate system visualization | Shows 3D axes clearly | Less interesting visually |
| Arrow (`'a'`) | Directional indicators | Clear pointing direction | Medium complexity |
| Cone (`'c'`) | Simple 3D shapes | Easy to recognize | Requires many points for smoothness |
| Tetrahedron (`'t'`) | Minimal 3D object | Simplest 3D shape | Limited visual appeal |

### Extension Techniques

| **Technique** | **Use Case** | **Advantages** | **Limitations** |
|---------------|--------------|----------------|-----------------|
| Chessboard AR | Calibration-based AR | High accuracy, easy calibration | Requires visible chessboard |
| ArUco Markers | Robust tracking | Unique IDs, occlusion-resistant | Requires pre-printed markers |
| OpenGL Rendering | Complex 3D models | Realistic lighting, high quality | Higher computational cost |
| Static Image AR | Offline processing | No real-time constraints | Not interactive |
| Homography | Planar surfaces | Fast, no camera calibration needed | Limited to 2D planes |

---

## Troubleshooting

### Common Issues

#### Issue: Chessboard not detected
**Symptoms:** No corners found, corners window shows red crosses

**Possible Causes & Solutions:**
1. **Incorrect chessboard size**
   - System expects 9×6 internal corners
   - Verify your printed chessboard matches
   - Update `pointsPerRow` and `pointsPerColumn` parameters if different

2. **Poor lighting**
   - Ensure even lighting without harsh shadows
   - Avoid glare on chessboard surface
   - Try adjusting exposure settings

3. **Motion blur**
   - Hold chessboard steady
   - Ensure camera is in focus
   - Reduce motion during capture

4. **Chessboard too small/large**
   - Move closer or farther from camera
   - Aim for chessboard filling 30-60% of frame

5. **Wrinkled or curved chessboard**
   - Use rigid, flat chessboard
   - Mount on cardboard or foam board

#### Issue: Poor calibration (high reprojection error)
**Symptoms:** Reprojection error > 1.0 pixels, virtual objects misaligned

**Solutions:**
1. **Capture more images** - Aim for 15-20 images instead of 5
2. **Increase angle variation** - Tilt and rotate chessboard to cover ±60° angles
3. **Cover entire frame** - Place chessboard in all regions (corners, edges, center)
4. **Check chessboard quality** - Ensure squares are perfectly flat and square
5. **Verify image sharpness** - Discard blurry calibration images
6. **Re-calibrate from scratch** - Delete `cameraParams.csv` and start fresh

#### Issue: Virtual objects don't align with chessboard
**Symptoms:** Objects drift, float above/below board, incorrect orientation

**Solutions:**
1. **Verify camera parameters** - Check `cameraParams.csv` exists and has reasonable values
2. **Ensure proper target size** - Object points assume unit square size, verify scaling
3. **Check pose estimation** - Look for solvePnP warnings in console
4. **Validate camera matrix** - fx and fy should be ~1000-2000 for typical webcams
5. **Test with static image** - Use extensions to debug with consistent input

#### Issue: AR projection stutters or lags
**Symptoms:** Low FPS, choppy rendering

**Solutions:**
1. Reduce image resolution in video capture
2. Simplify virtual object (fewer points/lines)
3. Compile with optimization flags (`-O3`)
4. Use faster chessboard detection (reduce refinement iterations)
5. Skip frames (process every Nth frame)

#### Issue: "Unable to open the primary video device" error
**Symptoms:** Error -404, program exits immediately

**Solutions:**
1. Check camera permissions (macOS System Preferences → Security & Privacy → Camera)
2. Verify camera is not in use by another application
3. Try different camera index: `cv::VideoCapture(1)` or `(2)`
4. Test camera with `ls /dev/video*` (Linux) or Photo Booth (macOS)

#### Issue: OpenGL extension won't compile
**Symptoms:** Linker errors about GLFW, GLEW, Assimp

**Solutions:**
1. Install missing libraries (see DEVELOPMENT.md)
2. Update library paths in makefile
3. Verify OpenGL version support: `glxinfo | grep "OpenGL version"` (Linux)
4. Check GLFW window creation succeeds

### Error Codes

- **-404**: Video device not found or camera access denied
- **-1**: Invalid chessboard size or detection failure
- **Assertion failed**: Camera parameters file missing or corrupted

---

## Dataset Organization

### Resource Files

```
resources/
├── cameraParams.csv      # Calibration results (generated)
├── checkerboard.png      # 9×6 chessboard pattern image
├── staticImage_1.jpg     # Test image 1 for static AR
├── staticImage_2.jpg     # Test image 2 for static AR
├── staticImage_3.jpg     # Test image 3 for static AR
├── staticImage_4.jpg     # Test image 4 for static AR
├── staticVid_1.mp4       # Test video 1 for static AR
└── staticVid_2.mp4       # Test video 2 for static AR
```

### Calibration Data Format

**cameraParams.csv:**
```csv
cameraMatrix,fx,0,cx,0,fy,cy,0,0,1
distCoeff,k1,k2,p1,p2,k3
reprojectionError,error_value
```

**Example:**
```csv
cameraMatrix,1303.0435,0.0000,184.4379,0.0000,1383.8424,103.2846,0.0000,0.0000,1.0000
distCoeff,-0.3816,10.6530,-0.0511,0.0079,-87.8130
reprojectionError,0.3992
```

**Parameter Interpretation:**
- **fx = 1303.04**: Horizontal focal length in pixels
- **fy = 1383.84**: Vertical focal length in pixels (slightly different due to non-square pixels)
- **cx = 184.44**: Principal point X (image center X)
- **cy = 103.28**: Principal point Y (image center Y)
- **k1 = -0.38**: Radial distortion (barrel/pincushion)
- **k2 = 10.65, k3 = -87.81**: Higher-order radial distortion
- **p1 = -0.05, p2 = 0.008**: Tangential distortion (lens misalignment)
- **Reprojection Error = 0.40 pixels**: Excellent calibration quality

### Recommended Setup

- **Chessboard size**: 9 columns × 6 rows (internal corners)
- **Square size**: 2.5-3.0 cm for typical webcam distances
- **Print quality**: High-resolution printer (300+ DPI), no scaling
- **Mounting**: Rigid backing (cardboard, foam board, or plexiglass)

---

## Extensions Implemented

This project includes several advanced extensions beyond the base requirements:

1. **ArUco Marker Tracking (`multiTarget.cpp`)**
   - Detects and tracks ArUco markers (DICT_6X6_250)
   - Supports multiple concurrent markers
   - More robust than chessboard in cluttered environments
   - Enables unique marker IDs for different AR content

2. **OpenGL 3D Model Rendering (`openGLExtension.cpp`)**
   - Hardware-accelerated rendering using OpenGL 3.3 core profile
   - Loads complex 3D models (Suzanne from Blender) using Assimp library
   - Real-time lighting and shading with programmable shaders
   - Perspective projection using GLM mathematics library
   - Requires: GLFW (window management), GLEW (extension loading), GLM (math), Assimp (model loading)

3. **Static Image/Video AR (`extensions.cpp`)**
   - Apply AR to static images: `./extensions <params> i <image> <object>`
   - Apply AR to videos: `./extensions <params> v <video> <object>`
   - Supports modes: `l` (live), `i` (static image), `v` (static video)
   - Useful for offline processing and demonstrations

4. **Homography-based Projection (`homographyExtension.cpp`)**
   - Alternative to PnP for planar surface tracking
   - Computes 2D-to-2D transformation (homography matrix)
   - Faster than full 3D pose estimation
   - Limited to flat surfaces parallel to image plane

5. **Harris Corner Detection (`features.cpp`)**
   - Real-time Harris corner detector
   - Visualizes detected corners with circles
   - Useful for feature tracking and SLAM applications
   - Adjustable parameters: block size, aperture, k-value

---

## Future Enhancements

Potential improvements could include:

- **Deep Learning-based Tracking** - CNN-based target detection for robustness
- **SLAM Integration** - Simultaneous Localization and Mapping for markerless AR
- **Occlusion Handling** - Depth sensing to hide virtual objects behind real objects
- **Multi-user AR** - Shared AR experiences across multiple devices
- **Dynamic Virtual Objects** - Interactive objects responsive to user input
- **Texture Mapping** - Apply images/videos as textures on virtual objects
- **Shadow Rendering** - Realistic shadows for virtual objects
- **Web-based Interface** - Browser-based AR using WebGL

---

## Notes

- Camera intrinsic parameters are device-specific - recalibrate for each camera
- Lighting conditions affect chessboard detection - consistent lighting improves reliability
- Virtual object scale is relative to chessboard square size (1 unit = 1 square)
- Distortion coefficients are crucial for wide-angle cameras (k1, k2 handle barrel distortion)
- Pose estimation accuracy degrades when chessboard is viewed at extreme angles (>60°)

---

## License

This project is part of academic coursework for CS-5330 at Northeastern University.

**Authors:**
- Samavedam Manikhanta Praphul
- Poorna Chandra Vemula

**Course:** CS-5330 Pattern Recognition and Computer Vision

### 📚 Usage as Reference

This repository is intended as a **learning resource and reference guide**. If you're working on a similar project:

- Use it to understand algorithm implementations and approaches
- Reference it when debugging your own code or stuck on concepts
- Learn from the structure and design patterns

Please respect academic integrity policies at your institution. This code should guide your learning, not replace it. Write your own implementations and cite references appropriately.

---

## Acknowledgments

- **Professor Bruce Maxwell** for guidance on camera calibration and pose estimation techniques
- **OpenCV community** for comprehensive computer vision library and documentation
- **CS-5330 course staff** for project specifications and calibration tutorials
- **Blender Foundation** for Suzanne 3D model used in OpenGL extension
- **ArUco library** for robust marker tracking implementation

### References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Pose Estimation (solvePnP)](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Harris Corner Detector](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html)
- [Zhang's Camera Calibration Method (1998)](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)
