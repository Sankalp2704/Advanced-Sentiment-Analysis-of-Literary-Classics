# Advanced Sentiment Analysis of Literary Classics: Comparative Study of Shakespeare and Melville Using Modern NLP

**Authors:** A Aakash, C Sai Sankalp, Daksh Nirwal
**Date:** November 11, 2025

## 📚 Overview

This repository contains the code and methodology for an interdisciplinary project investigating the **computational sentiment analysis** across three major literary classics:
1.  **William Shakespeare's *Romeo and Juliet***
2.  **William Shakespeare's *Hamlet***
3.  **Herman Melville's *Moby-Dick***

We employ a multi-layered approach using traditional lexicon-based methods (VADER, custom lexicons) and state-of-the-art transformer-driven models (DistilBERT) to quantify emotional arcs, compare model performance on complex literary texts, and critically assess the limitations of current NLP techniques relative to human literary interpretation.

## 🎯 Objectives

* To quantitatively map the **sentiment trajectories** across the segmented texts.
* To compare the effectiveness of **lexicon-based** (VADER) vs. **transformer-based** (DistilBERT) sentiment analysis approaches.
* To evaluate different **segmentation strategies** (scene/chapter vs. fixed-length chunking) and their impact on analysis.
* To identify distinctive sentiment patterns and correlations with narrative elements.

## 🛠️ Methodology and Implementation

The project is structured around three primary Python code pipelines, implemented in Google Colaboratory for reproducibility.

### Text Sources and Preprocessing
* **Source:** Literary texts are sourced from Project Gutenberg.
* **Preprocessing:** Includes stripping headers/footers, normalization of whitespace, and removal of extraneous characters[cite: 111, 114, 115].
* **Segmentation:** Three strategies are explored:
    1.  [cite_start]**Scene/Chapter-Based:** Uses explicit ACT/SCENE headings for plays and chapter titles for the novel[cite: 119, 120].
    2.  [cite_start]**Fixed-Length Chunking:** Overlapping blocks of approximately **2000 words with 200-word overlaps** for transformer model input[cite: 121, 122].
    3.  [cite_start]**Hybrid:** Combining semantic cues with fixed chunking[cite: 123].

### Sentiment Analysis Pipelines

| Pipeline | Texts Analyzed | Model(s) Used | Key Features |
| :--- | :--- | :--- | :--- |
| **Pipeline 1** | *Romeo and Juliet*, *Moby-Dick* | **VADER** and custom fallback lexicons | [cite_start]Lexicon-based sentiment scoring and segment-level tables[cite: 127, 128, 129]. |
| **Pipeline 2** | *Romeo and Juliet*, *Moby-Dick* | **VADER** and custom fallback lexicons | [cite_start]Extends P1 with enhanced visualization (histograms, smoothed trajectories) and word-level sentiment associations[cite: 130, 131, 132]. |
| **Pipeline 3** | *Romeo and Juliet*, *Hamlet* | **DistilBERT** (Hugging Face SST-2) | [cite_start]Transformer-based classification utilizing overlapping chunks for contextual sensitivity[cite: 134, 135, 136]. |

### Sentiment Scoring Metrics
[cite_start]Each segment produces four scores using VADER or aggregated transformer output[cite: 139, 140, 141, 143, 145]:
* **Positive (pos)**
* **Negative (neg)**
* **Neutral (neu)**
* **Compound (normalized aggregate score)**

## 💻 Prerequisites

The code is implemented in Python and requires the following key libraries. [cite_start]Specific versions used in the project are listed below for reproducibility[cite: 168, 169]:

```bash
# Key Python Libraries and Versions
numpy==1.21.2
pandas==1.3.3
matplotlib==3.4.3
vaderSentiment==3.3.2
transformers==4.10.2
