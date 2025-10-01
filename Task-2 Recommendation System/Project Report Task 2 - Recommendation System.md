# Task 2 Readme

# Group Member

1155194253 Huang Tsz Wing

1155192350 Tse Wing Yan

# Overview

is a Python implementation of a **Matrix Factorization-based Recommendation System** using **Stochastic Gradient Descent (SGD).** This script is designed to predict user-item ratings based on the training set and evaluation the model performance, then generate the prediction for the testing set. The implementation is using the SVD approach in collaborative filtering.

# Requirements

## Python Version

- Python 3.12.7 or higher

## Dependencies

- Numpy
- Pandas

# Input Data

1. Training Data (train.csv)
    
    The CSV file contains:
    
    - user_id
    - item_id
    - score
    
2. Testing Data
    
    The CSV file contains:
    
    - user_id
    - item_id
    
    The program will predict the score for user_id to item_id
    

# Output

The Predictions on the testing data (submit.csv) with the following format:

- user_id
- item_id
- predict_score (can be decimal)

# How It Works

## 1. Data Preparation

- The script reads the training and testing datasets.
- Maps user_id and item_id to internal indices for matrix operations

## 2. Model Initialization

Initialize:

- Latent factor matrices for user ($P$) and items ($Q$)
- Bias vector for user ($b_u$) and item ($b_i$)
- Global average rating ($\mu$).

## 3. Training

- Uses Stochastic Gradient Descent to learn latent factors and biases by minimizing the error between actual and predicted rating on the initial training data set.
- Prediction formula : $r_{xi} = \mu + b_x + b_i + p_x^\top q_i$
- Then update the user biases, item biases, latent factors based on the gradient of the loss function.

## 4. Prediction

- Predicts rating for testing dataset using the learned model parameters
- Handel missing users/items by defaulting their predictions to the global mean ($\mu$)

## 5. Output

- Save the predicted scores for the testing dataset in submit.csv