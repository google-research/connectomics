# Connectomics

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### _Note: Under heavy development_

_This repository is currently under significant development, and should not be considered stable. Semantic versioning is not guaranteed for now; APIs may - and likely will - change often and without notice._

## About this repository

Common code for analyzing volumetric datasets for connectomic reconstruction, including geometric primitives like `BoundingBox` and `BoxGenerator`, and dealing with segmentation ID ranges, including relabeling, making ranges contiguous, and testing whether two volumes contain equivalent sequences. 

Coming soon: pipeline primitives to process subvolumes.

## Projects using Connectomics

1. [SOFIMA](https://www.github.com/google-research/sofima) - SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment) is a tool for stitching, aligning and warping large 2d, 3d and 4d microscopy datasets.


_This is not an officially supported Google product._