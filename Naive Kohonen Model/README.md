# Bitcoin Price Prediction using Naive Bayes and Kohonen's SOM

This repository contains the code for predicting Bitcoin prices using a Naive Bayes model and Kohonen's Self-Organizing Map (SOM). The model also utilizes a weighted ensemble approach to combine predictions from both models.

## Overview

The goal of this project is to predict the future price direction (Up or Down) of Bitcoin based on historical data. The project employs a Naive Bayes model and Kohonen's SOM, with predictions combined using a weighted ensemble method.

## Dataset

The dataset used is `BTC-USD.csv`, which includes historical Bitcoin prices. The data is preprocessed, scaled, and split into training and testing sets.

## Model Architecture

### Naive Bayes Model
The Naive Bayes model is a probabilistic classifier that assumes feature independence.

### Kohonen's SOM
Kohonen's Self-Organizing Map is an unsupervised learning algorithm that projects high-dimensional data into lower dimensions (typically 2D) while preserving the topological structure of the data.

### Weighted Ensemble
The predictions from the Naive Bayes model and SOM are combined using a weighted ensemble approach to improve the accuracy.

## Mean Absolute Percentage Error (MAPE)

MAPE is used as the evaluation metric to compare the model's predictions with the actual labels.

## Usage

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. Install Jupyter Notebook or Upload the files to colab and run it

## Results

The model's performance is evaluated using MAPE for each model (Naive Bayes, SOM) and the combined predictions.

## Acknowledgements

This project is based on research in machine learning techniques for financial forecasting.
