#python script to get MNIST handwritten digits and preprocess
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images and normalize pixel values
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse=False, categories='auto')
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

# Split into train and validation sets
X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)

# Save data to text files
np.savetxt("X_train.txt", X_train)
np.savetxt("X_val.txt", X_val)
np.savetxt("y_train.txt", y_train_onehot)
np.savetxt("y_val.txt", y_val_onehot)
