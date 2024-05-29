# Comparison of Bitcoin Price Prediction Models: Stochastic Activation Function vs. Hybrid Naive Bayes and Kohonen's SOM

This repository contains the implementations of two distinct models for Bitcoin price prediction, based on two academic papers. The project is a part of the Deep Neural Network course, aiming to compare and contrast the methodologies and results of the two approaches.

## Overview

The objective of this project is to predict Bitcoin prices using two different approaches:
1. A neural network with a custom stochastic activation function.
2. A hybrid model combining Naive Bayes and Kohonen's Self-Organizing Map (SOM).

## Papers

### 1. Neural Network with Stochastic Activation Function
**Title**: Stochastic Neural Networks for Cryptocurrency Price Prediction
**Publisher**: [IEEE](https://ieeexplore.ieee.org/abstract/document/9079491)
**Description**: This paper introduces stochastic activation functions to neural networks to improve generalization. The stochasticity allows the model to capture more nuanced patterns in the data.

### 2. Hybrid Naive Bayes and Kohonen's SOM
**Title**: Hybrid Prediction Model for Financial Markets Using Naive Bayes and Kohonen's SOM
**Publisher**: [ArXives]()
**Description**: This paper presents a hybrid approach combining Naive Bayes and Kohonen's SOM for financial market prediction, leveraging the strengths of both techniques to improve prediction accuracy.

## Implementations

### Neural Network with Stochastic Activation Function

#### Dataset
The dataset used is `BTC-USD.csv`, which includes historical Bitcoin prices, downloaded from `Yahoo Finance`. The data is preprocessed and scaled before being fed into the neural network.

#### Model Architecture
- Dense layer with 150 units, ReLU activation
- Dense layer with 130 units, ReLU activation
- Dense layer with 100 units, ReLU activation
- Dense layer with 50 units, ReLU activation
- Dense layer with 25 units, ReLU activation
- Dense layer with 10 units, ReLU activation
- StochasticActivation layer
- Dense layer with 1 unit, linear activation

#### Custom Stochastic Activation Layer

##### Noise Function
```python
def noise_func(shape):
    return tf.random.normal(shape=shape, mean=0.0, stddev=0.6)
```

##### Reaction Function
```python
def reaction_func(ht, prev_state):
    return ht - prev_state
```

##### StochasticActivation Class
```python
class StochasticActivation(tf.keras.layers.Layer):
    def __init__(self, gamma, reaction_func, noise_func, activation="relu", **kwargs):
        super(StochasticActivation, self).__init__(**kwargs)
        self.gamma = gamma
        self.reaction_func = reaction_func
        self.activation = tf.keras.activations.get(activation)
        self.noise_func = noise_func 

    def call(self, inputs, prev_state=None):
        if prev_state is None:
            prev_state = tf.zeros_like(inputs)
        ht = self.activation(inputs)
        xi_t = self.noise_func(tf.shape(inputs))
        st = ht + self.gamma * xi_t * self.reaction_func(ht, prev_state)
        return st
```

#### Training
The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function for 120 epochs with a batch size of 32.

#### Evaluation
The model's performance is evaluated using the Mean Absolute Percentage Error (MAPE) and the absolute percentage difference between the predicted and actual prices.

### Hybrid Naive Bayes and Kohonen's SOM

#### Dataset
The dataset used is `BTC-USD.csv`, which includes historical Bitcoin prices, downloaded from `Yahoo Finance`. The data is preprocessed and scaled before being fed into the neural network.

#### Model Architecture

##### Naive Bayes
The Naive Bayes model is trained on the scaled closing prices to classify the price movement as "Up" or "Down".

##### Kohonen's SOM
A SOM is trained to cluster the scaled price data, and the cluster centers are used to predict price movements.

#### Training and Evaluation
Both models are trained separately, and their predictions are combined using a weighted ensemble approach. The Mean Absolute Percentage Error (MAPE) is calculated for each model and the ensemble.

#### Usage
##### Install Dependencies:

```bash
pip install -r requirements.txt
```

##### Run the Neural Network Script:
Install Jupyter Notebook or upload files to colab

##### Run the Hybrid Model Script:
Install Jupyter Notebook or upload files to colab

#### Results
The results include the trend of the price, and the accuracy calculated based on the mean absolute percentage difference for each model.

## Conclusion
This project demonstrates two different approaches to Bitcoin price prediction, highlighting the strengths and weaknesses of each method. The stochastic activation function introduces beneficial randomness to neural networks, while the hybrid model leverages the strengths of both Naive Bayes and SOM.

## Acknowledgements
This project is based on research in neural networks and machine learning techniques for financial forecasting.