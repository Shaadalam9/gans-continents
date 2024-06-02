# gans-continents

This repository contains the implementation of a hybrid Progressive GAN (ProGAN) and Conditional GAN (cGAN) with Transformer integration for generating high-quality video frames. The model is designed to handle temporal sequences in video data and progressively increase the resolution of generated images during training.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The hybrid GAN model implemented in this repository leverages the advantages of Progressive GANs and Conditional GANs, with added Transformer components for better handling of temporal dependencies in video data. The model is trained to generate realistic video frames conditioned on input data, such as location labels and counts of vehicles, trucks, and pedestrians.

## Architecture

The architecture consists of the following key components:

1. **Data Loader**: Loads video frames, location labels, and counts of vehicles, trucks, and pedestrians.
2. **Generator**:
   - **Input Preparation**: Combines noise, labels, and counts.
   - **Transformer Encoder**: Processes the combined input.
   - **Progressive Blocks**: Sequential convolutional layers that increase the resolution of generated images.
   - **To RGB Layers**: Converts generated features to RGB images at each resolution step.
3. **Discriminator**:
   - **From RGB Layer**: Converts input RGB images to feature maps.
   - **Progressive Blocks**: Sequential convolutional layers that decrease the resolution of input images.
   - **Combine Features, Labels, Counts**: Merges feature maps with labels and counts.
   - **Transformer Encoder**: Processes the combined input.
   - **Fully Connected Layer**: Outputs the validity score.
4. **Training Loop**: Manages the training process, including forward pass, loss computation, backpropagation, and optimization.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
