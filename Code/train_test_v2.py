import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

# 輸出資料夾,
output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/classification_output"
os.makedirs(output_dir, exist_ok=True)

# 載入 Digits 資料,
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int32)

# 標準化與降維（選前 4 維特徵）,
X = StandardScaler().fit_transform(X)
X = X[:, :4]

# one-hot label,
y_onehot = tf.keras.utils.to_categorical(y, num_classes=10)

# 切分訓練/測試集,
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 建立模型,
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,), name='input'),
    tf.keras.layers.Dense(4, activation='linear', name='dense1'),
    tf.keras.layers.Activation('relu', name='relu'),
    tf.keras.layers.Dense(10, activation='linear', name='dense2'),
    tf.keras.layers.Activation('softmax', name='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 評估效果,
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 儲存模型,
model_path = os.path.join(output_dir, 'model.h5')
model.save(model_path)
print(f"✅ Model saved to {model_path}")

# 儲存範例測試資料,
np.savetxt(os.path.join(output_dir, 'example_input.txt'), X_test[0], fmt="%.6f")
with open(os.path.join(output_dir, 'example_label.txt'), 'w') as f:
    f.write(str(np.argmax(y_test[0])))