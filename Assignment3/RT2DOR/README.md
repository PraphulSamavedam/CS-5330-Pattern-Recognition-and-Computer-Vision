# Assignment 3: Real-Time 2D Object Recognition (RT2DOR)

## Overview

This project implements a **Real-Time 2D Object Recognition** system that detects, segments, and classifies objects in both static images and live video streams. The system uses computer vision techniques for image segmentation combined with K-Nearest Neighbors (K-NN) classification to recognize objects based on their geometric and spatial features.

The system recognizes **17 different object classes** and achieves **83% average accuracy** on the test dataset. It provides multiple executables for different use cases: training classifiers, analyzing static images, detecting multiple objects, generating confusion matrices, and performing real-time video recognition.

**🎥 Demo Video:** [Watch the system in action](https://youtu.be/ZORyRWK4H6c)

## RT2DOR System Architecture

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

## Features

### Core Capabilities

1. **Training System** - Automated feature extraction and database creation
2. **Real-time Recognition** - Live video object detection and classification
3. **Static Image Analysis** - Detailed analysis of single images with intermediate steps
4. **Multiple Object Detection** - Simultaneous recognition of multiple objects in a scene
5. **Evaluation Tools** - Confusion matrix generation and accuracy metrics
6. **Morphological Processing** - Custom erosion, dilation, and region growing algorithms

### Feature Extraction Techniques

The system extracts 9-dimensional feature vectors for each object:

1. **Area Ratio** - Normalized object area relative to image size
2. **Aspect Ratio** - Width-to-height ratio of oriented bounding box
3. **Percent Filled** - Ratio of object pixels to bounding box area
4. **Hu Moments (7 values)** - Rotation, scale, and translation invariant shape descriptors
   - φ₁: Captures general shape characteristics
   - φ₂: Measures elongation
   - φ₃: Captures triangularity
   - φ₄-φ₇: Higher-order shape characteristics

### Classification Methods

- **K-Nearest Neighbors (K-NN)**: Configurable K value (default K=5)
- **Distance Metrics**:
  - Euclidean distance
  - Scaled Euclidean distance (weighted features, recommended)

### Supported Object Classes

The system is trained to recognize **17 different object types**:

1. Beanie
2. Book
3. Bottle Cap
4. Cap
5. Chess Board
6. Ear Buds
7. Fire Alarm
8. Glasses Case
9. Glove
10. Mask
11. Microphone (Mic)
12. Pen
13. Phone
14. Remote
15. Spoon
16. Umbrella
17. Wallet
18. Watch

**Unknown Object Handling:** When an object doesn't match any trained class (distance exceeds threshold), the system prompts the user to provide a new label and adds it to the training database.

## Quick Start

### Requirements
- **OpenCV 4.x** - For image processing and video capture
- Pre-compiled executables available in `bin/` folder (Windows)
- For building from source, see [DEVELOPMENT.md](DEVELOPMENT.md)

### Using Pre-compiled Executables (Windows)

The system comes with pre-compiled executables in the `bin/` directory:
- `train.exe` - Training program
- `Match.exe` - Real-time video recognition
- `MultipleObjects.exe` - Multiple object detector
- `StaticImageAnalyzer.exe` - Static image analysis
- `ConfusionMatrixGenerator.exe` - Evaluation tool

**Note:** OpenCV DLLs must be in your system PATH or in the same directory as the executables.

### Building from Source

For building from source (Linux/macOS/Windows), see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Usage

### 1. Training the Classifier

First, you need to train the system by providing labeled images:

**Windows:**
```bash
bin\train.exe <trainingImagesFolder> [csvFilePath] [mode]
```

**Linux/macOS:**
```bash
./train <trainingImagesFolder> [csvFilePath] [mode]
```

**Parameters:**
- `trainingImagesFolder` - Directory containing training images
- `csvFilePath` (optional) - Output CSV file (default: `../data/db/features.csv`)
- `mode` (optional) - `a` for automatic labeling from filename (default), `m` for manual labeling

**Training Modes:**
- **Automatic Mode** (default): Labels are extracted from filenames (e.g., `Beanie05.jpg` → label: "Beanie")
- **Manual Mode**: System displays each image and prompts user to enter a label (cannot be empty)

**Automatic Mode Example:**
```bash
# Windows
bin\train.exe data\train_images

# Linux/macOS
./train data/train_images
```

Filename format for automatic labeling: `ObjectName###.jpg` (e.g., `Beanie01.jpg`, `Cup05.jpg`)

**Manual Mode Example:**
```bash
# Windows
bin\train.exe data\train_images data\db\features.csv m

# Linux/macOS
./train data/train_images data/db/features.csv m
```
The system will prompt you to enter a label for each image.

**Training Process:**

The training system processes each image in the training folder:
1. **Load Image** - Read image from the training directory
2. **Extract Features** - Compute the 9-dimensional feature vector (preprocessing, segmentation, moment calculation)
3. **Obtain Label**:
   - **Automatic Mode**: Extract label from filename (e.g., `Beanie05.jpg` → "Beanie")
   - **Manual Mode**: Display image to user and prompt for label input (validated to be non-empty)
4. **Store to Database** - Append filepath, label, and feature vector to `features.csv`

This CSV file is then used during recognition to calculate distances without reprocessing images.

**Output:** Creates `features.csv` with format:
```
filepath,label,area_ratio,aspect_ratio,percent_filled,hu1,hu2,hu3,hu4,hu5,hu6,hu7
```

Example entry:
```
..\data\train_images/Beanie1.jpg,Beanie,0.8750,0.6968,0.7728,4.3103,3.0308,5.3249,-9.6972,-8.0480,9.6168
```

### 2. Real-time Video Recognition

Recognize objects in live video feed:

**Windows:**
```bash
bin\Match.exe
```

**Linux/macOS:**
```bash
./vidDisplay
```

**Controls:**
- Press `q` to quit
- Press `s` to save current frame
- System displays recognized label with K-NN confidence

**Configuration:** Edit `vidDisplay.cpp` to adjust:
- `K` value (line 27) - Number of nearest neighbors (default: 5)
- `grayscaleThreshold` (line 22) - Binary threshold (default: 124)
- `numberOfErosions` (line 23) - Morphological cleaning (default: 5)

### 3. Multiple Object Detection

Detect and recognize multiple objects in a static image:

**Windows:**
```bash
bin\MultipleObjects.exe <imagePath> <featuresFile> <distanceMetric> [K]
```

**Linux/macOS:**
```bash
./multipleObjects <imagePath> <featuresFile> <distanceMetric> [K]
```

**Parameters:**
- `imagePath` - Path to input image
- `featuresFile` - Path to trained features CSV (e.g., `data/db/features.csv`)
- `distanceMetric` - `euclidean` or `scaled_euclidean`
- `K` (optional) - Number of nearest neighbors (default: 1)

**Example:**
```bash
# Windows
bin\MultipleObjects.exe data\example001.png data\db\features.csv euclidean 3

# Linux/macOS
./multipleObjects data/example001.png data/db/features.csv euclidean 3
```

**Output:** Displays image with bounding boxes and labels for each detected object.

**Unknown Object Detection:** If a detected object's distance to the nearest training sample exceeds a predefined threshold, it's classified as "Unknown". The system then:
1. Prompts the user to enter a new label for the unknown object
2. Automatically appends the feature vector and new label to `features.csv`
3. Makes the object available for future recognition

This enables continuous learning and expansion of the recognition system without retraining.

### 4. Static Image Analysis

Analyze a single image with detailed intermediate steps:

**Windows:**
```bash
bin\StaticImageAnalyzer.exe <imagePath>
```

**Linux/macOS:**
```bash
./staticImageAnalyzer <imagePath>
```

**Example:**
```bash
# Windows
bin\StaticImageAnalyzer.exe data\example068.png

# Linux/macOS
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

**Windows:**
```bash
bin\ConfusionMatrixGenerator.exe <featuresFile> [outputCSV] [K]
```

**Linux/macOS:**
```bash
./confusionMatrix <featuresFile> [outputCSV] [K]
```

**Parameters:**
- `featuresFile` - Path to features CSV
- `outputCSV` (optional) - Output confusion matrix file
- `K` (optional) - Number of nearest neighbors (default: 1)

**Example:**
```bash
# Windows
bin\ConfusionMatrixGenerator.exe data\db\features.csv data\db\confusion_matrix.csv 3

# Linux/macOS
./confusionMatrix data/db/features.csv data/db/confusion_matrix.csv 3
```

**Output:** Confusion matrix CSV file showing predicted vs actual classifications.

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

## How It Works

### System Pipeline

#### Phase 1: Training (Offline)

1. **Load Training Images** - Read labeled images from directory
2. **Preprocessing** - Threshold to binary, clean with morphology
3. **Segmentation** - Identify object regions using region growing
4. **Feature Extraction** - Compute 9D feature vector per object
5. **Database Storage** - Save features and labels to CSV

#### Phase 2: Recognition (Real-time or Static)

1. **Capture Input** - Video frame or static image
2. **Preprocessing Pipeline**:
   - Threshold to binary (grayscale < 124)
   - Dilation (fill small holes)
   - Erosion (remove noise)
3. **Segmentation**:
   - Region growing (8-connected)
   - Select top N regions by area
4. **Feature Extraction** - Compute same 9D vector
5. **K-NN Classification**:
   - Load training database from features.csv
   - Compute distances to all training samples using scaled Euclidean distance
   - Select K=5 nearest neighbors
   - Majority vote for predicted label (with tie-breaking by smallest average distance)
6. **Display Results** - Draw bounding box with label

## Key Algorithms Explained

### 1. Thresholding

**Purpose:** Separate foreground objects from background.

**Algorithm:**
```
For each pixel (x,y):
    if grayscale_value(x,y) < threshold:
        binary(x,y) = 255  (white - foreground)
    else:
        binary(x,y) = 0    (black - background)
```

**Default Threshold:** 124 (middle of 0-255 range)

**Assumption:** Objects are darker than background (common for lab conditions)

#### Otsu's Automatic Thresholding (Extension)

**Purpose:** Automatically determine optimal threshold value instead of using fixed threshold.

**Algorithm:**
```
1. Compute histogram of grayscale intensities
2. For each possible threshold t (0 to 255):
   - Separate pixels into two classes: C₀ (< t) and C₁ (≥ t)
   - Compute class probabilities: w₀, w₁
   - Compute class means: μ₀, μ₁
   - Compute inter-class variance: σ²(t) = w₀ × w₁ × (μ₀ - μ₁)²
3. Optimal threshold = argmax(σ²(t))
```

**Advantage:** Adapts to varying lighting conditions, no manual tuning needed.

**Implementation:** Built from scratch based on Shapiro textbook reference.

### 2. Grass Fire Algorithm (Distance Transform)

**Purpose:** Compute distance of each pixel from background, used for erosion/dilation.

**Algorithm:**
```
1. Initialize: foreground = max_value, background = 0
2. Iteratively propagate distances:
   For each pixel:
       if pixel is foreground:
           distance = 1 + min(neighbor_distances)
3. Continue until convergence (no changes)
```

**Connectivity:** 4-connected (N, S, E, W) or 8-connected (includes diagonals)

**Use Case:** Foundation for morphological operations.

### 3. Erosion and Dilation

#### Erosion
**Purpose:** Shrink foreground regions, remove small noise.

**Algorithm:**
```
Using Grass Fire distances:
For each pixel:
    if distance(pixel) < numberOfErosions:
        pixel = background
```

**Effect:** Removes noise, separates touching objects, shrinks boundaries.

#### Dilation
**Purpose:** Expand foreground regions, fill small holes.

**Algorithm:**
```
1. Invert image (background becomes foreground)
2. Apply erosion on inverted image
3. Invert back
```

**Effect:** Fills holes, connects nearby regions, expands boundaries.

**Typical Sequence:** Dilation → Erosion (closing operation) removes noise while preserving size.

### 4. Region Growing (Connected Components)

**Purpose:** Identify separate connected objects in binary image.

**Algorithm:**
```
Using stack-based region growing:
1. Initialize region_id = 1
2. Scan image for unlabeled foreground pixel
3. When found:
   - Push pixel to stack
   - While stack not empty:
       - Pop pixel
       - Label with region_id
       - Push unlabeled neighbors to stack
   - Increment region_id
4. Repeat until all foreground pixels labeled
```

**Connectivity:** 8-connected (diagonal neighbors included)

**Output:** Region map where each pixel = region ID (1, 2, 3, ...)

**Use Case:** Separates multiple objects in same image.

### 5. Feature Extraction

#### Moments Calculation

**Spatial Moments:**
```
M_pq = Σ Σ (x^p)(y^q)   for all pixels in region
```

**Central Moments (translation invariant):**
```
μ_pq = Σ Σ (x - x̄)^p (y - ȳ)^q
where x̄ = M₁₀/M₀₀, ȳ = M₀₁/M₀₀
```

#### Geometric Features

1. **Area Ratio:**
```
area_ratio = region_pixel_count / total_image_pixels
```

2. **Aspect Ratio:**
```
aspect_ratio = oriented_bbox_width / oriented_bbox_height
```

3. **Percent Filled:**
```
percent_filled = region_pixel_count / (bbox_width × bbox_height)
```

#### Hu Moment Invariants

**Purpose:** Rotation, scale, and translation invariant shape descriptors.

**Formulas:** Computed from normalized central moments:
```
η_pq = μ_pq / (μ₀₀)^((p+q)/2 + 1)

φ₁ = η₂₀ + η₀₂
φ₂ = (η₂₀ - η₀₂)² + 4η₁₁²
φ₃ = (η₃₀ - 3η₁₂)² + (3η₂₁ - η₀₃)²
... (7 total invariants)
```

**Properties:**
- **φ₁, φ₂**: Represent overall shape compactness
- **φ₃**: Measures triangularity/skewness
- **φ₄-φ₇**: Higher-order shape characteristics

**Use Case:** These features remain constant under rotation, scaling, and translation, making them ideal for object recognition.

### 6. K-Nearest Neighbors (K-NN) Classification

**Purpose:** Predict object label based on similarity to training samples.

**Algorithm:**
```
1. Extract features from unknown object → query_vector
2. Load all training samples → database
3. For each training sample:
       distance = euclidean_distance(query_vector, sample_vector)
4. Sort by distance (ascending)
5. Select K nearest neighbors
6. Predicted_label = majority_vote(K_neighbor_labels)
```

**Euclidean Distance:**
```
distance = sqrt(Σ(query[i] - sample[i])²)
```

**Scaled Euclidean Distance (weighted):**
```
distance = sqrt(Σ w[i] × (query[i] - sample[i])²)
where w[i] = weights for each feature
```

**Choosing K:**
- **K=1**: Sensitive to noise, but fast
- **K=3**: Good balance, reduces noise impact
- **K=5**: Default value used in this system, robust to outliers
- **K=7+**: More stable, but slower and may oversmooth

**Tie-breaking Rule:** If multiple labels have the same vote count (e.g., 2 votes for "Book" and 2 votes for "Phone"), the system selects the label with the **smallest average distance** from the query object.

**Tie-breaking:** If multiple labels have same vote count, select label with smallest average distance.

### 7. Top N Segments Selection

**Purpose:** Focus on largest/most significant objects, ignore tiny noise regions.

**Algorithm:**
```
1. Count pixels per region → region_sizes
2. Sort regions by size (descending)
3. Optional: Apply minimum area threshold (total_area / 1000)
4. Select top N regions
5. Create new binary image with only these N regions
```

**Minimum Area Restriction:**
```
For each region:
    if region_size < (total_foreground_pixels / 1000):
        discard region
```

**Use Case:** In cluttered scenes, focus on significant objects and ignore specks of noise.

## Algorithm Workflow Example

### Complete Pipeline for Single Object Recognition

**Input:** Image of a coffee cup on white background

**Step 1: Threshold**
```
Original: RGB image (480×640)
↓ [grayscale conversion + threshold < 124]
Binary: Cup = white (255), Background = black (0)
```

**Step 2: Morphology (Clean)**
```
Binary → Dilation (5 iterations, 8-connected)
↓ [fills small holes inside cup]
Dilated → Erosion (5 iterations, 4-connected)
↓ [shrinks back to original size, noise removed]
Cleaned Binary Image
```

**Step 3: Region Growing**
```
Cleaned Binary → Region Growing (8-connected)
↓ [stack-based connected component labeling]
Region Map: Cup labeled as region_id = 1
```

**Step 4: Feature Extraction**
```
Region 1 (Cup):
├─ Compute moments: M₀₀, M₁₀, M₀₁, M₂₀, M₁₁, M₀₂, ...
├─ Calculate central moments: μ₂₀, μ₁₁, μ₀₂, ...
├─ Compute Hu moments: φ₁, φ₂, ..., φ₇
├─ Extract geometric features:
│  ├─ Area ratio = 0.152
│  ├─ Aspect ratio = 0.87
│  └─ Percent filled = 0.68
└─ Feature vector = [0.152, 0.87, 0.68, -4.23, -8.15, -11.3, -10.2, -20.1, -13.4, -25.6]
```

**Step 5: K-NN Classification (K=3)**
```
Training Database (features.csv):
├─ Cup samples: 15 entries
├─ Beanie samples: 12 entries
├─ Phone samples: 10 entries
└─ Pen samples: 8 entries

Distance Computation:
├─ distance(query, Cup01) = 0.15
├─ distance(query, Cup02) = 0.18
├─ distance(query, Beanie01) = 0.89
├─ distance(query, Cup03) = 0.22
├─ distance(query, Phone01) = 1.24
├─ ...

K=3 Nearest Neighbors:
├─ Cup01 (distance = 0.15)
├─ Cup02 (distance = 0.18)
└─ Cup03 (distance = 0.22)

Majority Vote: 3/3 = "Cup"
Prediction: Cup (100% confidence)
```

**Step 6: Display**
```
Draw oriented bounding box around cup
Place label "Cup" at top of bounding box
Display confidence score
```

## Performance Considerations

### Training Phase
- **Time Complexity**: O(N × M) where N = number of images, M = pixels per image
- **Typical Time**: 100 images (~2-5 minutes)
- **Bottleneck**: Region growing and moment computation

### Recognition Phase
- **Time Complexity**: O(K × D) where K = training samples, D = feature dimensions
- **Real-time Performance**:
  - Single object: ~30 FPS (33ms per frame)
  - Multiple objects (3-5): ~15 FPS (66ms per frame)
- **Bottleneck**: K-NN distance computation for large databases

### System Accuracy

The system achieves **83% average accuracy** across 17 object classes using K=5 and scaled Euclidean distance.

**Per-Class Accuracy (Top Performers):**
- Glove: 96.4%
- Beanie: 92.3%
- Mask: 91.7%
- Chess Board: 90.0%
- Ear Buds: 89.3%
- Remote: 89.5%
- Spoon: 88.9%
- Book: 88.2%

**Challenging Classes:**
- Cap: 65.0% (often confused with Fire Alarm, Wallet)
- Bottle Cap: 69.2% (confused with Cap, Book)
- Fire Alarm: 60.0% (confused with Cap, Book)
- Watch: 70.0% (confused with Mic)

**Key Observations:**
- Classes with distinctive shapes (Glove, Beanie, Mask) achieve >90% accuracy
- Similar-shaped objects (Cap vs Bottle Cap) show confusion
- Training with 10-20 samples per class, using scaled Euclidean distance

### Optimization Tips

1. **Reduce Database Size**: Only keep diverse, representative samples
2. **Limit K Value**: K=1 or K=3 is much faster than K=7+
3. **Lower Resolution**: Scale down input images (e.g., 640×480 → 320×240)
4. **Optimize Morphology**: Reduce erosion/dilation iterations if acceptable
5. **Compile with Optimization**: Use `-O3` flag for 20-30% speedup

## Choosing Algorithms and Parameters

### When to Use Different K Values

| **K Value** | **Best For** | **Pros** | **Cons** |
|-------------|--------------|----------|----------|
| K=1 | Small, clean datasets | Fast, simple | Sensitive to noise/outliers |
| K=3 | General purpose | Balanced, good noise reduction | Slightly slower |
| K=5-7 | Large, noisy datasets | Robust to outliers | Slower, may oversmooth |

### Distance Metrics

| **Metric** | **Formula** | **Use Case** |
|------------|-------------|--------------|
| Euclidean | `√Σ(a-b)²` | Equal feature importance |
| Scaled Euclidean | `√Σw(a-b)²` | Weighted features (e.g., prioritize Hu moments) |

**Recommendation:** Start with Euclidean and K=3, adjust based on confusion matrix results.

### Threshold Selection

- **Dark objects on light background**: threshold = 100-130
- **Light objects on dark background**: threshold = 150-180 or invert image
- **Mixed lighting**: May need adaptive thresholding (not implemented)

### Morphology Parameters

- **High noise images**: More erosion iterations (5-7)
- **Clean images**: Fewer iterations (2-3)
- **Touching objects**: Higher erosion can separate them
- **Small holes in objects**: More dilation iterations

## Troubleshooting

### Common Issues

#### Issue: Objects not detected
**Symptoms:** No bounding boxes shown, empty region map

**Possible Causes & Solutions:**
1. **Wrong threshold value**
   - Check if objects are darker/lighter than background
   - Try adjusting `grayscaleThreshold` (default: 124)
   - Visualize binary image to verify segmentation

2. **Too much erosion**
   - Object completely eroded away
   - Reduce `numberOfErosions` parameter

3. **Object too small**
   - Falls below minimum area threshold
   - Disable `minAreaRestriction` or adjust threshold

#### Issue: Multiple objects merged
**Symptoms:** Two objects labeled as one

**Solutions:**
1. Increase number of erosions to separate touching objects
2. Adjust `numberOfSegments` to allow more regions
3. Check if objects are actually touching in image

#### Issue: Poor classification accuracy
**Symptoms:** Objects frequently mislabeled

**Solutions:**
1. **Increase training samples** - Need 10-15 samples per class minimum
2. **Adjust K value** - Try K=3 or K=5 instead of K=1
3. **Check feature quality**:
   - Visualize Hu moments to ensure distinctiveness
   - Verify training images are properly labeled
4. **Use confusion matrix** to identify problematic classes
5. **Consider scaled Euclidean** if some features more important

#### Issue: Real-time recognition too slow
**Symptoms:** Low FPS, laggy video

**Solutions:**
1. Reduce input resolution in `vidDisplay.cpp`
2. Lower K value (K=1 is fastest)
3. Reduce database size (fewer training samples)
4. Compile with optimization flags (`-O3`)
5. Reduce erosion/dilation iterations

#### Issue: "File not found" errors
**Solutions:**
1. Check relative paths: `../data/db/features.csv`
2. Verify working directory when running executable
3. Use absolute paths if needed

### Error Codes

- **-100**: Image file not found
- **-404**: Missing required arguments or video device unavailable
- **-1000**: CSV file read error

## Dataset Organization

### Training Images Folder Structure

```
data/
├── train_images/
│   ├── Beanie01.jpg
│   ├── Beanie02.jpg
│   ├── Cup01.jpg
│   ├── Cup02.jpg
│   ├── Phone01.jpg
│   └── Pen01.jpg
├── test_images/
│   ├── test_beanie.jpg
│   └── test_multiple.jpg
└── db/
    ├── features.csv        (generated by training)
    └── confusion_matrix.csv (generated by evaluation)
```

### Recommended Training Set

- **Recommended samples per class**: 10-20 images per object type
- **Variety**: Different orientations, scales, lighting conditions
- **Image quality**: Clear, single object per image, clean background
- **Naming convention** (automatic mode): `ClassName##.jpg` (e.g., `Beanie01.jpg`, `Cup15.jpg`)
- **Background**: Use consistent white or light background for best results

## Extensions Implemented

This project includes several advanced extensions beyond the base requirements:

1. **Extended Object Recognition (17 Classes)**
   - Recognizes 17 different objects instead of the required 10
   - Includes diverse categories: wearables (Beanie, Cap, Glove, Mask, Watch), electronics (Phone, Mic, Remote, Ear Buds, Fire Alarm), stationery (Book, Pen, Wallet), and household items (Spoon, Umbrella, Glasses Case, Bottle Cap, Chess Board)

2. **Custom Morphological Operations from Scratch**
   - Implemented Grass Fire Transform (distance transform algorithm)
   - Built custom erosion and dilation algorithms using Grass Fire
   - Both 4-connected and 8-connected support
   - No reliance on OpenCV morphological functions

3. **Custom Segmentation Algorithms**
   - Region growing algorithm implemented from scratch (stack-based)
   - Union-find algorithm for connected component labeling
   - Support for both 4-connected and 8-connected neighborhood

4. **Otsu's Automatic Thresholding**
   - Implemented Otsu's method from scratch for automatic binary threshold selection
   - Calculates optimal threshold by maximizing inter-class variance
   - Based on Shapiro textbook reference

5. **Multiple Object Detection with Unknown Object Handling**
   - Detects and classifies multiple objects simultaneously in static images
   - Identifies unknown objects (objects not in training database)
   - Prompts user for new labels and dynamically updates training database
   - Distance threshold-based unknown detection

## Future Enhancements

Potential improvements:
- **Deep Learning Features**: Replace hand-crafted features with CNN embeddings
- **Color Features**: Add color histograms for better discrimination
- **Texture Features**: Gabor filters or LBP for textured objects
- **Adaptive Thresholding**: Handle varying lighting conditions
- **Online Learning**: Continuous learning from user corrections

## Notes

- Feature vectors are scale and rotation invariant (Hu moments)
- System assumes objects darker than background
- Best results with high-contrast images
- Morphological parameters may need tuning per dataset

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

## Acknowledgments

- **Professor Bruce Maxwell** for debugging assistance with connected component analysis (union-find) and discussions on computing oriented bounding boxes
- **OpenCV community** for computer vision capabilities and comprehensive documentation
- **CS-5330 course staff** for guidance and project specifications

### References

- [Otsu Thresholding with OpenCV](https://learnopencv.com/otsu-thresholding-with-opencv/)
- [OpenCV Thresholding Tutorial](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [OpenCV Moments Tutorial](https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html)
- [Shape Matching using Hu Moments](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)
