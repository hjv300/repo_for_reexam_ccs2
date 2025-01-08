# Multimodal Song Topic Classification

This repository contains the code and resources for the project in the Computational Cognitive Science 2 exploring the use of multimodal data (lyrics and audio features) to classify song topics. The project compares two models:
- A **Baseline BERT Classifier** that uses only song lyrics.
- A **Multimodal BERT Classifier** that combines lyrics and audio features.

## Project Overview

The goal of this project is to evaluate the effectiveness of integrating textual and audio features for song topic classification. By leveraging a combination of BERT for lyrics and audio feature analysis.

## Features

- **Baseline Model**: Fine-tunes a pretrained BERT model on song lyrics for topic classification.
- **Multimodal Model**: Combines BERT embeddings with normalized audio features for improved classification performance.
- **Dataset Preprocessing**: Includes cleaning, tokenization, and normalization of input data.
- **Performance Evaluation**: Reports metrics such as accuracy, precision, recall, and F1-score.

## Installation

Clone the repository:
```bash
git clone https://github.com/hjv300/repo_for_reexam_ccs2.git
cd repo_for_reexam_ccs2 
```

Install dependencies:
```bash
pip install -r requirements.txt
```
To run the models do:
```bash
python3 BertUnimodal.py 
python3 BertMultimodal.py
```
