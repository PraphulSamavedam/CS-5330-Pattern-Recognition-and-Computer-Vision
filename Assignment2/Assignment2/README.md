# Assignment 2: Content-Based Image Retrieval (CBIR) System

## Overview

This project implements a Content-Based Image Retrieval (CBIR) system that finds and displays similar images from a database based on visual features. The system extracts various features from images (color histograms, texture, spatial distributions) and uses distance metrics to find the most similar images to a target query image.

The system provides both a **GUI-based interface** and a **command-line interface** for flexibility in usage.

## CBIR System Query Flow

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

## Features

### Feature Extraction Techniques

The system implements multiple feature extraction techniques:

1. **Baseline (9x9)** - Extracts center 9x9 pixel values as features
2. **2D Histogram (rg chromaticity)** - Normalized histogram in rg chromaticity space
3. **3D Histogram (RGB)** - Normalized 3D histogram in RGB color space
4. **Multi-Histogram (Upper-Bottom)** - Separate histograms for top and bottom halves
5. **Multi-Histogram (Left-Right)** - Separate histograms for left and right halves
6. **Texture & Color (TAC)** - Combines gradient magnitude and color histograms
7. **Q4 Texture Histogram** - Uses 4 quarters plus texture information
8. **Custom Histogram** - Center region color with texture features (65-35% weighting)

Additional specialized techniques:
- Specific color detection (Yellow/Banana, Blue bins, Green bins)
- Edge and color histogram combination
- Two halves with rg chromaticity

### Distance Metrics

Multiple distance metrics are available to compare feature vectors:

1. **Sum of Squared Error (AggSquareError)** - Aggregate squared differences
2. **Histogram Intersection Error** - Measures histogram overlap
3. **Entropy Error** - Compares entropy differences between feature vectors
4. **Weighted Histogram Error (W82)** - Weighted (80-20) histogram comparison
5. **Mean Square Error** - Root mean square of differences
6. **Masked Boundary Error** - Specialized for masked/segmented images

## Quick Start

### Requirements
- OpenCV 4.x
- C++ compiler with C++11 support
- See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed build instructions

### Building
For detailed build instructions, dependencies, and troubleshooting, see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Usage

### GUI Interface

```bash
./guiMain <targetImagePath> <imagesDatabasePath>
```

**Example:**
```bash
./guiMain data/testImg/pic.0164.jpg data/olympus
```

**GUI Workflow:**
1. Program opens Feature Settings window
2. Select a feature extraction technique from available buttons
3. Optional: Check "Re-evaluate ft vectors" to recompute all features
4. Program opens Distance & Images Settings window
5. Use trackbar to select number of similar images (3-15)
6. Select a distance metric from available buttons
7. Results are displayed showing target image and similar images
8. Press 'q' to quit or 's' to save results

### Command-Line Interface

```bash
./cmdMain <targetImagePath> <imagesDatabasePath> <featureTechnique> <distanceMetric> <numberOfSimilarImages> \
          [resetFile] [echoStatus]
```

**Parameters:**
- `targetImagePath` - Path to the query/target image
- `imagesDatabasePath` - Directory containing the image database
- `featureTechnique` - Feature extraction method:
  - `Baseline` - 9x9 baseline
  - `2DHistogram` - rg chromaticity histogram
  - `3DHistogram` - RGB histogram
  - `2HalvesUBHistogram` - Upper-Bottom multi-histogram
  - `2HalvesLRHistogram` - Left-Right multi-histogram
  - `TACHistogram` - Texture and color
  - `Q4TextureHistogram` - 4-quarters with texture
  - `CustomHistogram` - Center color with texture
- `distanceMetric` - Distance calculation method:
  - `AggSquareError` - Sum of squared error
  - `HistogramError` - Histogram intersection
  - `EntropyError` - Entropy difference
  - `W82HistogramError` - Weighted 80-20 histogram
  - `MeanSquareError` - Mean square error
  - `MaskedBoundError` - Masked boundary error
- `numberOfSimilarImages` - Number of similar images to retrieve (integer)
- `resetFile` (optional) - Set to non-zero to recalculate feature vectors
- `echoStatus` (optional) - Set to non-zero for verbose output

**Example:**
```bash
./cmdMain data/testImg/pic.0164.jpg data/olympus 3DHistogram HistogramError 5
```

This retrieves 5 similar images using 3D RGB histogram features and histogram intersection distance metric.

## How It Works

### 1. Feature Extraction
When you run the program for the first time with a specific feature technique, it:
- Scans all images in the database directory
- Computes feature vectors for each image
- Stores the feature vectors in a CSV file (e.g., `3DHistogram.csv`)
- Subsequent runs reuse this CSV file unless reset is requested

### 2. Distance Calculation
For the target image:
- Extracts features using the selected technique
- Compares against all database image features using the distance metric
- Ranks images by similarity (lower distance = more similar)

### 3. Result Display
- Displays the top N most similar images
- Shows distance values for each match
- Target image is displayed for reference

### 4. Saving Results
Press 's' key to save retrieved images with descriptive filenames containing:
- Feature technique used
- Distance metric used
- Result rank
- Original filename

## Performance Considerations

- **First run** with a feature technique processes all images and may take time
- **Subsequent runs** are fast as features are cached in CSV files
- **Larger databases** (like olympus with 1000+ images) benefit significantly from caching
- Use the **reset option** when database contents change

## Key Algorithms Explained

### 1. Color Histogram Algorithms

#### RGB 3D Histogram (`rgb3DHistogramTechnique`)
**Purpose:** Captures color distribution across the entire image in 3D color space.

**How it works:**
- Divides each color channel (R, G, B) into bins (typically 8 bins = 32 values per bin)
- For each pixel: `bin_index = pixel_value / bin_size`
- Creates 3D histogram: 8×8×8 = 512 bins
- Normalizes by dividing by total pixel count

**Formula:**
```
r_index = R / binSize
g_index = G / binSize
b_index = B / binSize
histogram[r_index][g_index][b_index]++
```

**Use case:** Finding images with similar overall color composition (e.g., sunset photos, blue ocean images).

#### rg Chromaticity Histogram (`rg2DHistogramTechnique`)
**Purpose:** Color matching independent of brightness/illumination.

**How it works:**
- Converts RGB to normalized rg space (removes intensity):
  - `r = R / (R+G+B+ε)` where ε prevents division by zero
  - `g = G / (R+G+B+ε)`
- Creates 2D histogram (16×16 = 256 bins by default)
- More robust to lighting changes than RGB

**Use case:** Finding objects with similar colors under different lighting conditions.

---

### 2. Texture Analysis Algorithms

#### Gradient Magnitude (`gradientAndColorHistApproach`)
**Purpose:** Captures edge and texture information using Sobel filters.

**How it works:**
- Applies Sobel X filter (horizontal edges):
  ```
  [-1  0  1]
  [-2  0  2]
  [-1  0  1]
  ```
- Applies Sobel Y filter (vertical edges):
  ```
  [-1 -2 -1]
  [ 0  0  0]
  [ 1  2  1]
  ```
- Computes magnitude: `sqrt(sobelX² + sobelY²)`
- Creates histogram of gradient magnitudes

**Use case:** Distinguishing smooth vs textured objects (glass vs fabric, sky vs trees).

#### Laplacian Edge Detection (`centerColorAndTextureApproach`)
**Purpose:** Detects edges and texture patterns using second derivative.

**How it works:**
- OpenCV's Laplacian filter detects rapid intensity changes
- Captures second derivative of image intensity
- Combined with color histograms for robust matching

**Formula:** `∇²I = ∂²I/∂x² + ∂²I/∂y²`

**Use case:** Finding images with similar edge patterns (buildings, geometric shapes).

---

### 3. Spatial Information Algorithms

#### Multi-Region Histograms (Upper-Bottom, Left-Right)
**Purpose:** Preserve spatial layout information while using color histograms.

**Upper-Bottom Split (`twoHalvesUpperBottomApproach`):**
- Divides image horizontally at midpoint
- Computes separate RGB histograms for top and bottom
- Feature vector = [top_histogram, bottom_histogram]
- **Use case:** Images with consistent layout (sky above, ground below)

**Left-Right Split (`twoHalvesLeftRightApproach`):**
- Divides image vertically at midpoint
- Preserves horizontal spatial layout
- **Use case:** Objects positioned left/right (person standing, two objects side-by-side)

#### Center-Weighted Approach (`centerColorAndTextureApproach`)
**Purpose:** Focus on central object, reduce background influence.

**How it works:**
- Extracts center region (25×25 pixels or percentage)
- Computes histogram with higher weight (65%)
- Adds texture information with lower weight (35%)

**Formula:** `distance = 0.65 × color_distance + 0.35 × texture_distance`

**Use case:** Objects typically centered in photos (portraits, product images).

#### Four-Quarter Approach (`quartersAndTextureApproach`)
**Purpose:** Preserve 2D spatial color distribution with fine granularity.

**How it works:**
- Divides image into 4 equal quadrants:
  ```
  [Top-Left]  [Top-Right]
  [Bot-Left]  [Bot-Right]
  ```
- Computes separate histogram for each quadrant
- Concatenates all 4 histograms

**Use case:** When spatial arrangement is important (faces, structured scenes).

---

### 4. Distance Metric Algorithms

#### Histogram Intersection (`histogramIntersectionError`)
**Purpose:** Measures overlap between two normalized histograms.

**Formula:**
```
intersection = sum(min(histogram1[i], histogram2[i]))
distance = 1 - intersection
```

**Properties:**
- Range: 0 (identical) to 1 (completely different)
- Robust to partial occlusion
- Works well with color histograms

**Best for:** Histogram-based features (3D RGB, rg chromaticity).

#### Sum of Squared Error - SSE (`aggSquareError`)
**Purpose:** Classic Euclidean distance squared between feature vectors.

**Formula:**
```
distance = sum((feature1[i] - feature2[i])²)
```

**Properties:**
- Sensitive to magnitude differences
- Simple and computationally fast
- Can be dominated by outliers

**Best for:** Baseline technique (raw pixel values), features with similar scales.

#### Entropy Error (`entropyError`)
**Purpose:** Measures information content difference between distributions.

**Formula:**
```
entropy(H) = -sum(H[i] × log(H[i]))
distance = |entropy(H1) - entropy(H2)|
```

**Properties:**
- Low entropy = uniform distribution (smooth images)
- High entropy = diverse distribution (textured images)
- Measures complexity similarity

**Best for:** Comparing texture complexity, pattern diversity.

#### Weighted Histogram Error (`weightedHistogramIntersectionError`)
**Purpose:** Allows different importance weights for combined features.

**Formula:**
```
distance = 1 - sum(weight[j] × intersection_error(section_j))
```

**Example:** 80% weight on color, 20% on texture

**Use case:** Combined features where some aspects matter more (color-dominant vs texture-dominant matching).

#### Mean Square Error - MSE (`meanSquaredError`)
**Purpose:** Normalized version of SSE, independent of feature vector length.

**Formula:**
```
distance = sqrt(mean((feature1[i] - feature2[i])²))
```

**Properties:**
- Normalizes by feature vector length
- Better for comparing different feature types
- Less sensitive to dimensionality

**Best for:** Comparing features of different sizes or mixed feature types.

#### Masked Boundary Error (`maskedBoundaryError`)
**Purpose:** Specialized for comparing segmented/masked regions.

**Formula:**
```
overlap = min(area1, area2) + min(ratio1, ratio2)
distance = 2 - overlap
```

**Properties:**
- Assumes 2 features: normalized contour area, aspect ratio
- Measures geometric similarity
- Useful for object recognition

**Best for:** Shape-based matching, segmented object comparison.

---

### 5. Baseline Technique Algorithm

#### Center Pixel Extraction (`baselineTechnique`)
**Purpose:** Simplest feature extraction - raw pixel values from image center.

**How it works:**
- Extracts n×n pixel values from image center (default 9×9 = 81 pixels)
- Stores in RGB order: 81 pixels × 3 channels = 243 features
- Direct comparison without statistical processing

**Formula:**
```
center_row = height / 2
center_col = width / 2
feature[i] = image[center_row - n/2 : center_row + n/2,
                   center_col - n/2 : center_col + n/2]
```

**Limitations:**
- Requires well-aligned images
- Sensitive to translation, rotation, scale
- No statistical summarization

**Use case:** Baseline for comparison; works when images are registered/aligned.

---

### 6. Feature Caching Algorithm

#### CSV-based Feature Storage
**Purpose:** Avoid recomputing features for large databases - cache for speed.

**How it works:**
1. **First run:**
   - Compute features for all images
   - Store as: `[filename, feature1, feature2, ..., featureN]` in CSV
2. **Subsequent runs:**
   - Load from CSV (milliseconds vs minutes)
   - Skip computation entirely
3. **When to reset:**
   - Database contents change
   - Different feature technique selected
   - User manually requests via `resetFile` flag

**Performance Impact:**
- **Without cache:** 1000 images × 200ms = 3.3 minutes
- **With cache:** Load CSV = 2 seconds
- **Speedup:** ~100x faster

**File format example:**
```csv
data/olympus/pic.0001.jpg,0.124,0.089,0.201,...
data/olympus/pic.0002.jpg,0.098,0.156,0.087,...
```

---

### Algorithm Combination Examples

#### Example 1: Texture + Color (TAC)
```
1. Compute gradient magnitude using Sobel filters
2. Create histogram of gradient (texture features)
3. Create RGB histogram of original image (color features)
4. Combine with equal weights: 50% texture + 50% color
```

#### Example 2: Custom Histogram (Center + Texture)
```
1. Extract center 25×25 region
2. Compute RGB histogram of center (weight: 65%)
3. Apply Laplacian to detect edges
4. Compute edge histogram (weight: 35%)
5. Combine: 0.65 × center_color + 0.35 × edge_texture
```

#### Example 3: Complete Query Process
```
Query Image: "red_car.jpg"
1. Extract 3D RGB histogram → 512 values
2. Load database features from "3DHistogram.csv" → 1000 images
3. Compute histogram intersection for each:
   - car_1.jpg: distance = 0.12 (very similar)
   - flower.jpg: distance = 0.89 (very different)
4. Sort by distance: [car_1, car_2, truck, ...]
5. Return top 5 matches
```

---

### Choosing the Right Combination

| **Scenario** | **Best Feature** | **Best Distance Metric** |
|--------------|------------------|--------------------------|
| Color-based objects (fruits, clothing) | 3D RGB Histogram | Histogram Intersection |
| Different lighting conditions | rg Chromaticity | Histogram Intersection |
| Texture-rich images (fabrics, patterns) | Gradient Magnitude | Entropy Error |
| Spatially consistent scenes (landscapes) | Upper-Bottom Histogram | Histogram Intersection |
| Centered objects (portraits) | Center-Weighted | Weighted Histogram |
| Raw pixel matching (aligned images) | Baseline 9×9 | Sum of Squared Error |

**General Rule:** Match feature type to query need, match distance metric to feature distribution.

## Notes

- Feature vectors are stored in CSV format for reusability
- Images must be in formats supported by OpenCV (JPG, PNG, BMP, etc.)
- The system handles images of varying sizes
- Memory usage scales with database size and feature dimensionality

## Troubleshooting

For common issues, error codes, and solutions, see the **[Troubleshooting section in DEVELOPMENT.md](DEVELOPMENT.md#troubleshooting)**.

Common errors:
- **Error -100**: Insufficient command-line arguments
- **Error -400**: Image too small for baseline technique
- **Error -1000**: Fewer images found than requested

## Future Enhancements

Potential improvements could include:
- Deep learning-based features
- Query refinement through relevance feedback
- Multi-modal queries (combine multiple target images)
- Real-time video query support
- Web-based interface

## License

This project is part of academic coursework for CS-5330 at Northeastern University.

**Author:** Samavedam Manikhanta Praphul
**Course:** CS-5330 Pattern Recognition and Computer Vision

### 📚 Usage as Reference

This repository is intended as a **learning resource and reference guide**. If you're working on a similar project:

- Use it to understand algorithm implementations and approaches
- Reference it when debugging your own code or stuck on concepts
- Learn from the structure and design patterns

Please respect academic integrity policies at your institution. This code should guide your learning, not replace it. Write your own implementations and cite references appropriately.

## Acknowledgments

- CVUI library for simple OpenCV-based GUI: https://dovyski.github.io/cvui/
- OpenCV community for image processing capabilities
