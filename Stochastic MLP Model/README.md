# Bitcoin Price Prediction using Stochastic Activation Function in Neural Networks

This repository contains the code for predicting Bitcoin prices using a neural network model with a custom stochastic activation function.

## Overview

The goal of this project is to predict future Bitcoin prices based on historical data using a neural network that includes a stochastic activation function layer. This layer introduces randomness into the activation function to potentially capture more nuanced patterns in the data.

## Dataset

The dataset used is `BTC-USD.csv`, which includes historical Bitcoin prices. The data is preprocessed and scaled before being fed into the neural network.

## Model Architecture

The model consists of several dense layers with ReLU activation functions, followed by a custom `StochasticActivation` layer. The architecture is as follows:
- Dense layer with 150 units, ReLU activation
- Dense layer with 130 units, ReLU activation
- Dense layer with 100 units, ReLU activation
- Dense layer with 50 units, ReLU activation
- Dense layer with 25 units, ReLU activation
- Dense layer with 10 units, ReLU activation
- StochasticActivation layer
- Dense layer with 1 unit, linear activation

## Custom Stochastic Activation Layer

### Noise Function
```python
def noise_func(shape):
    return tf.random.normal(shape=shape, mean=0.0, stddev=0.6)
```

### Reaction Function
```python
def reaction_func(ht, prev_state):
    return ht - prev_state
```

### StochasticActivation Class
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

## Training

The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function for 120 epochs with a batch size of 32.

## Evaluation

The model's performance is evaluated using the Mean Absolute Percentage Error (MAPE) and the absolute percentage difference between the predicted and actual prices.

## Usage

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
2. Install Jupyter Notebook or Upload the files to colab and run it

## Results
The model's predicted prices are compared against the actual prices, and the accuracy is calculated based on the absolute percentage difference.

## Acknowledgements
This project is based on research in neural networks and machine learning techniques for financial forecasting.