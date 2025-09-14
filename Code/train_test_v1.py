import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
import sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/default_output"
os.makedirs(output_dir, exist_ok=True)

digits = load_digits()
X = digits.data.astype(np.float32)
y = to_categorical(digits.target, num_classes=10)


x = StandardScaler().fit_transform(X)
x = PCA(n_components=4).fit_transform(x)
y = PCA(n_components=4).fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(4,), name='input'),
    layers.Dense(4, activation='linear', name='dense1'),
    layers.Activation('relu', name='relu'),
    layers.Dense(4, activation='linear', name='dense2')
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32)

model_path = os.path.join(output_dir, 'model.h5')
model.save(model_path)
print(f"Saved model as {model_path}")
