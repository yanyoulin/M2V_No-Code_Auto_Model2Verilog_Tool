import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ========== 輸出資料夾 ==========
output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/classification_output"
os.makedirs(output_dir, exist_ok=True)

# ========== 載入並處理 MNIST 資料 ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ========== 建立模型 ==========
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,), name='input'),
    #tf.keras.layers.Dense(32, activation='linear', name='dense1'),
    #tf.keras.layers.Activation('relu', name='relu1'),
    tf.keras.layers.Dense(10, activation='linear', name='dense2'),
    tf.keras.layers.Activation('softmax', name='softmax')
])

# ========== 編譯與訓練 ==========
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

# ========== 儲存模型與範例測資 ==========
model.save(os.path.join(output_dir, "model.h5"))
np.savetxt(os.path.join(output_dir, "example_input.txt"), x_test[0])
with open(os.path.join(output_dir, "example_label.txt"), "w") as f:
    f.write(str(np.argmax(y_test_cat[0])))

print("✅ 模型儲存成功，測試輸入與標籤已輸出。")
