import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv("train.csv", names=["user_id", "item_id", "score"])
test_data = pd.read_csv("test.csv", names=["user_id", "item_id"])

# Get the number of unique users and items
n_users = train_data['user_id'].nunique()
n_items = train_data['item_id'].nunique()

# Create mappings from IDs to indices
user_mapping = {x: i for i, x in enumerate(train_data['user_id'].unique())}
item_mapping = {i: j for j, i in enumerate(train_data['item_id'].unique())}
train_data['user_idx'] = train_data['user_id'].map(user_mapping)
train_data['item_idx'] = train_data['item_id'].map(item_mapping)


# Set hyperparameters
n_factors = 1200  # Number of latent factors
n_epochs = 30  # Number of epochs
alpha = 0.01  # Learning rate
lamb = 0.1  # Regularization parameter 

# Initialize latent factor matrices and biases
P = np.random.normal(scale=0.1, size=(n_users, n_factors))  # User latent factors
Q = np.random.normal(scale=0.1, size=(n_items, n_factors))  # Item latent factors
b_x = np.zeros(n_users)  # User biases
b_i = np.zeros(n_items)  # Item biases
mu = train_data['score'].mean()  # Global mean

# Stochastic Gradient Descent (SGD) to train the model
for epoch in range(n_epochs):
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        actual_rating = row['score']

        # Predict the rating
        pred_rating = mu + b_x[user_idx] + b_i[item_idx] + np.dot(P[user_idx], Q[item_idx])
        error = actual_rating - pred_rating

        # Update biases
        b_x[user_idx] += alpha * (error - lamb * b_x[user_idx])
        b_i[item_idx] += alpha * (error - lamb * b_i[item_idx])

        # Update latent factors
        P[user_idx] += alpha * (2*error * Q[item_idx] - 2*lamb * P[user_idx])
        Q[item_idx] += alpha * (2* error * P[user_idx] - 2*lamb * Q[item_idx])

    # Calculate RMSE on training data
    total_error = 0
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        actual_rating = row['score']
        pred_rating = mu + b_x[user_idx] + b_i[item_idx] + np.dot(P[user_idx], Q[item_idx])
        total_error += (actual_rating - pred_rating) ** 2
    
    train_rmse = np.sqrt(total_error / len(train_data))

    print(f"Epoch {epoch + 1}/{n_epochs}, Train RMSE: {train_rmse:.4f}")

# Map test data user_id and item_id to indices
test_data['user_idx'] = test_data['user_id'].map(user_mapping)
test_data['item_idx'] = test_data['item_id'].map(item_mapping)

# Handle users/items that are in test.csv but not in train.csv
test_data['user_idx'] = test_data['user_idx'].fillna(-1).astype(int)
test_data['item_idx'] = test_data['item_idx'].fillna(-1).astype(int)

# Predict scores for test data
def predict(row):
    user_idx = row['user_idx']
    item_idx = row['item_idx']

    if user_idx == -1 or item_idx == -1:  # Handle missing users/items
        return mu
    return mu + b_x[user_idx] + b_i[item_idx] + np.dot(P[user_idx], Q[item_idx])

test_data['predicted_score'] = test_data.apply(predict, axis=1)
test_data[['user_id', 'item_id', 'predicted_score']].to_csv("predictions.csv", index=False)
