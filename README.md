# PAPS Hybrid Vision-Audio Search System

This repository contains the implementation for the paper:  
**"PAPS: Performing Audio-Prompted Search with Multi-Modal Detection"**.  

This project presents a hybrid vision-audio system. By leveraging multimodal inputs and ergodic trajectory optimization, this system addresses the limitations of traditional methods, enabling audio-prompted search in zero shot environment.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)

---

## Overview

Hybrid vision-audio systems integrate visual and audio cues for robust search applications. This project focuses on:
- **Ergodic trajectory optimization**: Ensures systematic domain coverage proportional to the expected utility of visiting specific regions.
- **Multimodal input processing**: Combines audio and visual modality to improve search efficiency.


Key Use Cases:
- Disaster relief
- Security surveillance
- Eldercare assistance

---

# Usage

## Visual system

```bash
# Ensure your input data is structured as follows:
input_data/
├── depth_images/    
├── Image/                
output.npz   # Camera parameters 

# Update the Rootdir variable in run2.py to point to your input data directory:
Rootdir = "/path/to/input_data/"

# run 
python audio_match/run2.py

# result in 
audio_match/pipeline_output

# Plug x,y,z coordinates and confidence values into run/drone_try.py
# Modify args in drone_try.py as needed

# Run drone_try.py 
python3 drone_try.py #Expected output: folders with xyz data and depth & RBG images
