# Color Space Analysis

## Purpose
Analyze and compare different color spaces (RGB, HSV, Lab) for food image segmentation.

## Why Color Spaces Matter
- **RGB**: Standard representation, but not perceptually uniform
- **HSV**: Separates color (Hue) from intensity (Value), good for color-based segmentation
- **Lab**: Perceptually uniform, L channel for brightness, a/b for color

## What This Script Does
1. Converts images to RGB, HSV, and Lab color spaces
2. Generates histogram analysis for each channel
3. Demonstrates Lab normalization for contrast enhancement

## Run
```bash
python color_analysis.py
```

## Output
- Histogram comparisons saved to `outputs/01_color_representation/`
- Sample images in `images/` folder

## Selected Classes
Analyzes pizza and sushi as representative examples of different color profiles.
