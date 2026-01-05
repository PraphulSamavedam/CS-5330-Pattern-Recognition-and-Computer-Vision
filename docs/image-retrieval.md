# Content-Based Image Retrieval (CBIR) System

## Overview

The Content-Based Image Retrieval (CBIR) system is a sophisticated image search engine that finds and ranks similar images from a database based on visual features rather than text metadata. The system analyzes visual characteristics such as color distribution, texture patterns, and spatial layout to identify images with similar content.

The implementation provides both a **GUI-based interface** for interactive exploration and a **command-line interface** for automated processing and integration into workflows.

!!! info "Key Concept"
    Unlike traditional text-based image search, CBIR uses visual content analysis to find similar images. You query with an image, not keywords, making it ideal for visual search applications, duplicate detection, and content organization.

## System Architecture

The CBIR system follows a multi-stage pipeline that efficiently processes queries through feature extraction, distance computation, and result ranking:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONTENT-BASED IMAGE RETRIEVAL                      │
│                              Query Flow Diagram                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │ Query Image  │
                              │ (Target)     │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  Select Feature Technique      │
                    │  (Baseline, 3DHistogram, etc.) │
                    └────────────┬───────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────┐
                    │  Extract Target Features       │
                    │  → Feature Vector (e.g., 512D) │
                    └────────────┬───────────────────┘
                                 │
                                 |
                                 │
                                 ▼
                    ┌───────────────────────────────┐
                    │ Check Corresponding CSV Cache │
                    │ (E.g. 3DHistogram.csv)        │
                    └────────────┬──────────────────┘
                                 │
                                 │ Found?
                                 ├────────────Yes ────────────────┐
                                 │                                │
                                 No                               │
                                 │                                │
                                 │                                │
                      ┌───────────────────────┐                   │
                      │ Image Database        │                   │
                      │ (olympus/, testDB/)   │                   │
                      └──────────┬────────────┘                   │
                                 │                                │
                                 ▼                                |
                    ┌──────────────────────────┐                  │
                    | Process All DB Images    │                  │
                    │ ├─ Read Image            │                  │
                    │ ├─ Extract Features      │                  │
                    │ └─ Store in CSV          │                  │
                    └──────────┬───────────────┘                  │
                               │                                  |
                               └┬─────────────────────────────────┘
                      ┌──────────────────┐
                      │ Read Features    │
                      │ from CSV (fast!) │
                      └────────┬─────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Database Features Loaded        │
              │ (N images × Feature Dimensions) │
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │ Select Distance Metric          │
              │ (HistogramError, SSE, etc.)     │
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │ Compute Distances               │
              │ For Each DB Image:              │
              │   distance[i] = metric(         │
              │     target_features,            │
              │     db_features[i]              │
              │   )                             │
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │ Rank by Distance                │
              │ Sort ascending (lower = similar)│
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │ Select Top-K Similar Images     │
              │ (User specified: 3-15 images)   │
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │ Display Results                 │
              │ ├─ Target Image                 │
              │ ├─ Similar Image 1 (dist: 0.12) │
              │ ├─ Similar Image 2 (dist: 0.18) │
              │ └─ Similar Image K (dist: 0.25) │
              └──────────────┬──────────────────┘
                             │
                      ┌──────┴───────┐
                      │              │
                      ▼              ▼
            ┌──────────────┐   ┌─────────────┐
            │ Save Results │   │ Exit (q)    │
            │ Press 's'    │   │             │
            └──────────────┘   └─────────────┘

Performance Note: First Run = Minutes | Cached Runs = Seconds
```

### Pipeline Stages

1. **Feature Extraction**: Converts images into numerical feature vectors that capture visual characteristics
2. **Feature Caching**: Stores computed features in CSV files for ~100x speedup on subsequent queries
3. **Distance Computation**: Compares query image features against database using selected metric
4. **Result Ranking**: Sorts images by similarity (lower distance = more similar)
5. **Result Display**: Shows top-K most similar images with distance scores

## Key Features and Capabilities

### Core Capabilities

- **Multiple Feature Extraction Methods**: 8+ techniques for different visual characteristics
- **Flexible Distance Metrics**: 6 different similarity measures for various use cases
- **Intelligent Caching**: CSV-based feature storage reduces query time from minutes to seconds
- **Dual Interface**: Both GUI and command-line options for different workflows
- **Scalable Performance**: Handles databases with 1000+ images efficiently
- **Result Persistence**: Save retrieved images with descriptive filenames

### Feature Extraction Techniques

The system implements multiple complementary approaches to capture different aspects of visual similarity:

#### 1. **Baseline (9×9 Center Pixels)**

Extracts raw pixel values from the center 9×9 region of the image.

- **Vector Size**: 243 features (81 pixels × 3 RGB channels)
- **Best For**: Aligned/registered images where objects are consistently positioned
- **Limitations**: Sensitive to translation, rotation, and scale variations

**How it works:**
```
center_row = height / 2
center_col = width / 2
feature[i] = image[center_row - 4 : center_row + 5,
                   center_col - 4 : center_col + 5]
```

#### 2. **2D Histogram (rg Chromaticity)**

Uses normalized color distribution in rg chromaticity space, removing intensity information.

- **Vector Size**: 256 features (16×16 bins)
- **Best For**: Color matching under varying lighting conditions
- **Advantages**: Illumination invariant

**Formula:**
```
r = R / (R + G + B + ε)
g = G / (R + G + B + ε)
```

Where ε prevents division by zero.

#### 3. **3D Histogram (RGB Color Space)**

Captures full color distribution across all three color channels simultaneously.

- **Vector Size**: 512 features (8×8×8 bins)
- **Best For**: Finding images with similar overall color composition
- **Use Cases**: Sunset photos, ocean images, fruit images

**Binning:**
```
r_index = R / 32  (256 values → 8 bins)
g_index = G / 32
b_index = B / 32
histogram[r_index][g_index][b_index]++
```

#### 4. **Multi-Histogram (Upper-Bottom Split)**

Divides image horizontally and computes separate histograms for spatial awareness.

- **Vector Size**: 1024 features (2 × 512)
- **Best For**: Images with consistent vertical layout (sky/ground, head/body)
- **Preserves**: Vertical spatial relationships

#### 5. **Multi-Histogram (Left-Right Split)**

Divides image vertically to preserve horizontal spatial layout.

- **Vector Size**: 1024 features (2 × 512)
- **Best For**: Horizontally structured scenes, objects positioned left/right
- **Preserves**: Horizontal spatial relationships

#### 6. **Texture & Color (TAC)**

Combines gradient magnitude histograms with color histograms for rich feature representation.

- **Vector Size**: Variable (gradient histogram + color histogram)
- **Best For**: Distinguishing textured vs smooth surfaces
- **Components**: 50% Sobel gradient texture + 50% RGB color

**Gradient Computation:**
```
Sobel_X = [[-1, 0, 1],    Sobel_Y = [[-1, -2, -1],
           [-2, 0, 2],               [ 0,  0,  0],
           [-1, 0, 1]]               [ 1,  2,  1]]

magnitude = sqrt(Sobel_X² + Sobel_Y²)
```

#### 7. **Q4 Texture Histogram (Four Quarters)**

Divides image into four quadrants with texture information for fine-grained spatial awareness.

- **Vector Size**: 2048+ features (4 quadrants × 512 + texture)
- **Best For**: Spatially structured scenes (faces, layouts)
- **Preserves**: 2D spatial color distribution

**Quadrant Division:**
```
[Top-Left]    [Top-Right]
[Bottom-Left] [Bottom-Right]
```

#### 8. **Custom Histogram (Center-Weighted + Texture)**

Focuses on central region with weighted combination of color and edge features.

- **Vector Size**: Variable
- **Best For**: Centered objects (portraits, product images)
- **Weighting**: 65% center color + 35% Laplacian edge texture

**Distance Formula:**
```
distance = 0.65 × color_distance + 0.35 × texture_distance
```

#### Additional Specialized Techniques

- **Specific Color Detection**: Targeted detection for yellow/banana tones, blue bins, green bins
- **Edge and Color Combination**: Canny edges with color histograms
- **Two Halves with rg Chromaticity**: Spatial split using illumination-invariant colors

### Distance Metrics

The system provides multiple distance metrics to compare feature vectors, each suited for different feature types:

#### 1. **Sum of Squared Error (SSE) - `AggSquareError`**

Classic Euclidean distance squared between feature vectors.

```
distance = Σ(feature1[i] - feature2[i])²
```

- **Properties**: Simple, fast, magnitude-sensitive
- **Best For**: Baseline technique, features with similar scales
- **Range**: [0, ∞)

#### 2. **Histogram Intersection Error - `HistogramError`**

Measures overlap between normalized histograms.

```
intersection = Σ min(histogram1[i], histogram2[i])
distance = 1 - intersection
```

- **Properties**: Robust to partial occlusion, bounded output
- **Best For**: All histogram-based features (RGB, rg, multi-region)
- **Range**: [0, 1]

#### 3. **Entropy Error - `EntropyError`**

Compares information content and complexity of feature distributions.

```
entropy(H) = -Σ H[i] × log(H[i])
distance = |entropy(H1) - entropy(H2)|
```

- **Properties**: Measures distribution complexity similarity
- **Best For**: Texture complexity comparison, pattern diversity
- **Interpretation**: Low entropy = uniform/smooth, High entropy = diverse/textured

#### 4. **Weighted Histogram Error (80-20) - `W82HistogramError`**

Allows different importance weights for combined feature components.

```
distance = 1 - Σ weight[j] × intersection_error(section_j)
```

- **Properties**: Flexible component weighting
- **Best For**: Combined features (e.g., 80% color + 20% texture)
- **Use Case**: Color-dominant vs texture-dominant matching

#### 5. **Mean Square Error (MSE) - `MeanSquareError`**

Normalized version of SSE, independent of feature vector length.

```
distance = sqrt(mean((feature1[i] - feature2[i])²))
```

- **Properties**: Normalized by vector length, scale-independent
- **Best For**: Comparing different feature types, mixed features
- **Advantage**: Less sensitive to dimensionality differences

#### 6. **Masked Boundary Error - `MaskedBoundError`**

Specialized metric for comparing segmented regions and object shapes.

```
overlap = min(area1, area2) + min(ratio1, ratio2)
distance = 2 - overlap
```

- **Properties**: Geometric similarity measure
- **Best For**: Shape-based matching, segmented object comparison
- **Features**: Normalized contour area + aspect ratio

### Feature-Metric Matching Guide

| **Scenario** | **Recommended Feature** | **Recommended Metric** |
|--------------|------------------------|----------------------|
| Color-based objects (fruits, clothing) | 3D RGB Histogram | Histogram Intersection |
| Different lighting conditions | rg Chromaticity | Histogram Intersection |
| Texture-rich images (fabrics, patterns) | Texture & Color (TAC) | Entropy Error |
| Spatially consistent scenes (landscapes) | Upper-Bottom Histogram | Histogram Intersection |
| Centered objects (portraits, products) | Custom (Center-Weighted) | Weighted Histogram |
| Raw pixel matching (aligned images) | Baseline 9×9 | Sum of Squared Error |
| Object shape similarity | Any + segmentation | Masked Boundary Error |

## Benefits and Limitations

!!! success "Benefits"
    **Efficient Performance**

    - CSV caching provides ~100x speedup: first run in minutes, cached runs in seconds
    - Scalable to databases with 1000+ images
    - Parallel-friendly architecture for batch processing

    **Flexible Feature Extraction**

    - 8+ feature techniques capture different visual characteristics
    - Supports color, texture, spatial layout, and combined approaches
    - Illumination-invariant options (rg chromaticity)

    **Multiple Distance Metrics**

    - 6 distance functions for different similarity criteria
    - Histogram intersection robust to partial occlusion
    - Weighted combinations for multi-modal features

    **User-Friendly Interface**

    - Interactive GUI for exploration and experimentation
    - Command-line interface for automation and scripting
    - Visual result display with distance scores

    **Research and Learning**

    - Demonstrates fundamental computer vision techniques
    - Modular design for extending with new features
    - Educational value for understanding CBIR systems

!!! warning "Limitations"
    **Computational Constraints**

    - First-time feature extraction takes minutes for large databases
    - Memory usage scales with database size and feature dimensionality
    - Real-time performance requires pre-computed features

    **Feature Limitations**

    - Color histograms lose spatial information (except multi-region variants)
    - Texture features sensitive to image resolution and quality
    - Baseline technique requires object alignment

    **Invariance Challenges**

    - Most features not rotation or scale invariant
    - Significant viewpoint changes affect matching
    - Background clutter can interfere with object-focused queries

    **Database Requirements**

    - Requires manual cache reset when database contents change
    - No automatic change detection for database updates
    - CSV cache files must match current database state

    **Semantic Gap**

    - Visual similarity ≠ semantic similarity
    - Cannot understand high-level concepts (e.g., "happy person")
    - No relevance feedback or query refinement in current implementation

!!! tip "Optimization Strategies"
    - Use cached features whenever possible (avoid unnecessary `resetFile`)
    - Choose appropriate feature-metric combinations for your use case
    - Start with 3D Histogram + Histogram Intersection for general-purpose queries
    - Use rg chromaticity for illumination-robust color matching
    - Combine spatial techniques (multi-region) for layout-sensitive queries
    - Consider texture features when color alone is insufficient

## Usage Examples

### GUI Interface

The graphical interface provides an interactive workflow for exploring different feature techniques and distance metrics.

**Basic Command:**
```bash
./guiMain <targetImagePath> <imagesDatabasePath>
```

**Example:**
```bash
./guiMain data/testImg/pic.0164.jpg data/olympus
```

**Interactive Workflow:**

1. **Feature Selection Window** opens first
   - Click a button to select feature extraction technique
   - Options: Baseline, 2D Histogram, 3D Histogram, Upper-Bottom, Left-Right, TAC, Q4 Texture, Custom
   - Optional: Check "Re-evaluate ft vectors" to force feature recomputation

2. **Distance & Settings Window** opens next
   - Use trackbar to select number of similar images (3-15)
   - Click a button to select distance metric
   - Options: AggSquareError, HistogramError, EntropyError, W82HistogramError, MeanSquareError

3. **Results Display** shows:
   - Query/target image (reference)
   - Top-K similar images ranked by distance
   - Distance scores for each match

4. **Keyboard Controls:**
   - Press `s` to save retrieved images with descriptive filenames
   - Press `q` to quit the application

**Saved Filename Format:**
```
<FeatureTechnique>_<DistanceMetric>_<Rank>_<OriginalFilename>
```

Example: `3DHistogram_HistogramError_1_pic.0245.jpg`

### Command-Line Interface

The command-line interface enables scripting, automation, and batch processing.

**Basic Command:**
```bash
./cmdMain <targetImagePath> <imagesDatabasePath> <featureTechnique> \
          <distanceMetric> <numberOfSimilarImages> [resetFile] [echoStatus]
```

**Parameters:**

- `targetImagePath` - Path to the query/target image
- `imagesDatabasePath` - Directory containing the image database
- `featureTechnique` - Feature extraction method (see list below)
- `distanceMetric` - Distance calculation method (see list below)
- `numberOfSimilarImages` - Number of similar images to retrieve (integer)
- `resetFile` (optional) - Non-zero to force feature recomputation (default: 0)
- `echoStatus` (optional) - Non-zero for verbose output (default: 0)

**Feature Technique Options:**

- `Baseline` - 9×9 center pixel extraction
- `2DHistogram` - rg chromaticity histogram
- `3DHistogram` - RGB color histogram
- `2HalvesUBHistogram` - Upper-Bottom multi-histogram
- `2HalvesLRHistogram` - Left-Right multi-histogram
- `TACHistogram` - Texture and color combination
- `Q4TextureHistogram` - Four-quarters with texture
- `CustomHistogram` - Center-weighted color with texture

**Distance Metric Options:**

- `AggSquareError` - Sum of squared error (SSE)
- `HistogramError` - Histogram intersection
- `EntropyError` - Entropy difference
- `W82HistogramError` - Weighted 80-20 histogram
- `MeanSquareError` - Mean square error (MSE)
- `MaskedBoundError` - Masked boundary error

**Example 1: Basic Query**
```bash
./cmdMain data/testImg/pic.0164.jpg data/olympus 3DHistogram HistogramError 5
```

Retrieves 5 similar images using 3D RGB histogram features and histogram intersection metric.

**Example 2: Query with Feature Recomputation**
```bash
./cmdMain data/testImg/pic.0164.jpg data/olympus 3DHistogram HistogramError 5 1 0
```

Forces recomputation of all features (useful when database changed).

**Example 3: Verbose Query**
```bash
./cmdMain data/testImg/pic.0164.jpg data/olympus TACHistogram EntropyError 10 0 1
```

Retrieves 10 images using texture+color with verbose output for debugging.

**Example 4: Illumination-Robust Query**
```bash
./cmdMain data/testImg/sunset.jpg data/olympus 2DHistogram HistogramError 8
```

Uses rg chromaticity for lighting-invariant color matching.

**Example 5: Spatial-Aware Query**
```bash
./cmdMain data/testImg/landscape.jpg data/olympus 2HalvesUBHistogram HistogramError 7
```

Uses upper-bottom split to preserve vertical spatial layout (sky/ground).

### Batch Processing Script Example

```bash
#!/bin/bash
# Query all images in a directory and save results

TARGET_DIR="data/testImages"
DATABASE="data/olympus"
FEATURE="3DHistogram"
METRIC="HistogramError"
NUM_RESULTS=5

for img in "$TARGET_DIR"/*.jpg; do
    echo "Processing: $img"
    ./cmdMain "$img" "$DATABASE" "$FEATURE" "$METRIC" "$NUM_RESULTS" 0 1
done
```

## Learning Objectives

This project demonstrates fundamental concepts in computer vision and pattern recognition:

### 1. Feature Extraction Fundamentals

- **Color Representation**: Understanding RGB, rg chromaticity, and color spaces
- **Histogram Techniques**: Building and normalizing multi-dimensional histograms
- **Texture Analysis**: Gradient magnitude, Sobel filters, Laplacian edge detection
- **Spatial Encoding**: Preserving layout information through region-based features

### 2. Distance Metrics and Similarity

- **Euclidean Distance**: SSE and MSE for vector comparison
- **Histogram Comparison**: Intersection, entropy, and weighted methods
- **Metric Selection**: Matching distance functions to feature types
- **Normalization**: Handling features with different scales and ranges

### 3. System Design Principles

- **Caching Strategy**: Trading disk space for computation time (CSV storage)
- **Modular Architecture**: Separating feature extraction from distance computation
- **Interface Design**: Supporting both interactive (GUI) and automated (CLI) workflows
- **Scalability**: Handling databases from tens to thousands of images

### 4. Computer Vision Concepts

- **Content-Based Retrieval**: Visual similarity vs semantic similarity
- **Feature Trade-offs**: Discriminative power vs computational cost
- **Invariance Properties**: Illumination, rotation, scale, and translation robustness
- **The Semantic Gap**: Limitations of low-level features for high-level concepts

### 5. Practical Skills

- **OpenCV Usage**: Image I/O, color space conversion, filtering operations
- **Performance Optimization**: Feature caching, efficient data structures
- **Experimental Methodology**: Testing different feature-metric combinations
- **Result Analysis**: Interpreting distance scores and ranking quality

### 6. Algorithm Understanding

- **Convolution Operations**: Sobel and Laplacian filters for edge detection
- **Statistical Summaries**: Histograms as probability distributions
- **Dimensionality**: Trading feature vector size for discriminative power
- **Weighting Schemes**: Combining multiple feature types effectively

## Technical Details

### Performance Characteristics

**Feature Extraction Times** (approximate, 1000 images):

- Baseline: ~30 seconds (simple pixel extraction)
- 2D/3D Histogram: ~2-3 minutes (histogram computation)
- Texture features: ~4-5 minutes (gradient computation)
- Multi-region: ~3-4 minutes (multiple histograms)

**Query Times with Caching:**

- Feature loading from CSV: <2 seconds
- Distance computation (1000 images): <1 second
- Total cached query time: <3 seconds
- **Speedup factor**: ~100x faster than recomputation

**Memory Usage:**

- 3D Histogram (512 features): ~4KB per image
- 1000 images: ~4MB feature storage
- Multi-region techniques: ~8MB for 1000 images

### Feature Vector Dimensions

| Feature Technique | Vector Size | Storage per Image |
|------------------|-------------|------------------|
| Baseline (9×9) | 243 | ~1 KB |
| 2D Histogram (16×16) | 256 | ~1 KB |
| 3D Histogram (8×8×8) | 512 | ~2 KB |
| Upper-Bottom Split | 1024 | ~4 KB |
| Left-Right Split | 1024 | ~4 KB |
| TAC (Texture + Color) | ~600-800 | ~3 KB |
| Q4 Texture | ~2100 | ~8 KB |
| Custom (Center + Texture) | Variable | ~2-3 KB |

### CSV Cache Format

Features are stored in CSV format for efficient reuse:

```csv
filename,feature_0,feature_1,feature_2,...,feature_N
data/olympus/pic.0001.jpg,0.124,0.089,0.201,...,0.156
data/olympus/pic.0002.jpg,0.098,0.156,0.087,...,0.234
data/olympus/pic.0003.jpg,0.201,0.045,0.178,...,0.092
```

Each feature technique maintains its own CSV file:
- `3DHistogram.csv` - RGB histogram features
- `2DHistogram.csv` - rg chromaticity features
- `Baseline.csv` - Center pixel features
- And so on...

### Error Codes

- **Error -100**: Insufficient command-line arguments
- **Error -400**: Image too small for baseline technique (requires >9×9)
- **Error -1000**: Fewer images found in database than requested
- **Error -200**: Cannot open/read target image file
- **Error -300**: Invalid feature technique or distance metric specified

## Build and Requirements

### Dependencies

- **OpenCV 4.x** - Core image processing library
- **C++11 or later** - Standard library features
- **CVUI library** - Simple OpenCV-based GUI components

### Building the Project

For detailed build instructions, dependency installation, and troubleshooting, refer to:

**[Assignment 2 DEVELOPMENT.md](/Users/prapsama/Documents/Personal/CS-5330-Pattern-Recognition-and-Computer-Vision/Assignment2/Assignment2/DEVELOPMENT.md)**

Quick build overview:
```bash
# Navigate to project directory
cd Assignment2/Assignment2

# Compile GUI version
g++ -o guiMain guiMain.cpp features.cpp distance.cpp \
    `pkg-config --cflags --libs opencv4`

# Compile command-line version
g++ -o cmdMain cmdMain.cpp features.cpp distance.cpp \
    `pkg-config --cflags --libs opencv4`
```

### Supported Image Formats

The system supports all image formats handled by OpenCV:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- WebP (.webp)

Images of varying sizes are supported - the system handles resizing and normalization internally.

## Future Enhancements

Potential improvements and research directions:

**Deep Learning Integration**
- CNN-based feature extraction (VGG, ResNet features)
- Learned distance metrics
- Transfer learning from pre-trained models

**Advanced Querying**
- Relevance feedback and query refinement
- Multi-modal queries (combining multiple target images)
- Weighted region-of-interest selection
- Partial image matching

**Performance Optimization**
- GPU acceleration for feature extraction
- Approximate nearest neighbor search (LSH, FAISS)
- Incremental database updates
- Parallel batch processing

**Additional Features**
- Real-time video query support
- Web-based interface with REST API
- Mobile application integration
- Cloud-based deployment

**Enhanced Techniques**
- SIFT/SURF keypoint-based matching
- Object segmentation integration
- Multi-scale feature pyramids
- Attention-based feature weighting

---

[← Back to Home](index.md)
