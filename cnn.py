import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn dữ liệu
train_dir = r'C:\Users\Admin\PycharmProjects\task10_xlha\input\Train'
val_dir = r'C:\Users\Admin\PycharmProjects\task10_xlha\input\Validation'

# Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Lưu mô hình
model.save('dog_cat_classifier.h5')

# Vẽ đồ thị
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Dự đoán trên ảnh bất kỳ
def predict_image(image_path, model):
    image = load_img(image_path, target_size=(150, 150))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    class_names = ['Cat', 'Dog']
    return class_names[int(prediction[0] > 0.5)]

# Dùng mô hình để dự đoán
test_image_path = r'C:\Users\Admin\PycharmProjects\task10_xlha\test\test1.jpeg'
model = tf.keras.models.load_model('dog_cat_classifier.h5')
prediction = predict_image(test_image_path, model)
print(f"Dự đoán: {prediction}")
