# VanishingPointDetection
Identify straight lines that appear to converge toward a common vanishing point

# Vanishing Point Line Detector

This Python program detects straight lines in input images and highlights those likely to converge toward a vanishing point. It utilizes OpenCV for image processing and scikit-learn's DBSCAN for clustering similar line angles.

## Features

- Detects lines via Hough Transform  
- Groups nearly-parallel lines using DBSCAN  
- Filters out horizontal and vertical lines  
- Extends selected lines to the image borders  
- Saves step-by-step image results

## Requirements

- Python 3.x  
- OpenCV  
- NumPy  
- scikit-learn

Install dependencies with:

```bash
pip install opencv-python numpy scikit-learn
````

## Usage

1. Place input images in the `input/` directory (`.jpg`, `.png`, etc.).
2. Run the script:

```bash
python main.py
```

3. Output images will be saved in the `output/` folder:

* `detected_lines.jpg`: All detected lines
* `filtered_lines.jpg`: Lines selected by angle threshold
* `extended_lines.jpg`: Filtered lines extended across image
* `overlayed_extended_lines.jpg`: Extended lines overlayed on original image

## How It Works

1. Convert image to grayscale → blur → edge detection (Canny)
2. Detect lines with `HoughLinesP`
3. Calculate line angles and cluster them via DBSCAN
4. Remove horizontal/vertical lines, keeping those diagonally oriented
5. Extend those lines to image edges
6. Overlay results and optionally compute intersections

## File Structure

```
.
├── input/                    # Place input images here
├── output/                   # Result images are saved here
├── main.py                   # Main detection script
└── README.md
```

## Customization

* `minLineLength`, `maxLineGap`: Tune Hough Transform sensitivity
* `eps`, `min_samples`: Adjust DBSCAN clustering granularity
* `angle_threshold`: Control filtering of unwanted lines


