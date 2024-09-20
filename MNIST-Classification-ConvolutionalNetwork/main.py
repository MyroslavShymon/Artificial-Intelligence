import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Завантажуємо датасет MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Підготовка даних: масштабування і перетворення
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

# Перетворення міток в one-hot вектор
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Створення згорткової нейронної мережі
model = models.Sequential()

# Перший згортковий шар
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Другий згортковий шар
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Третій згортковий шар
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Перетворення даних у вектор
model.add(layers.Flatten())

# Додаємо повнозв'язний шар
model.add(layers.Dense(64, activation='relu'))

# Вихідний шар
model.add(layers.Dense(10, activation='softmax'))

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Виведення інформації про модель
model.summary()

# Навчання моделі
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Оцінка моделі на тестовому наборі даних
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Функція для завантаження і обробки зображення
def load_and_prepare_image(image_path):
    # Завантажуємо зображення
    img = Image.open(image_path).convert('L')  # Перетворення в градації сірого
    img = img.resize((28, 28))  # Зміна розміру до 28x28
    img_array = np.array(img)  # Перетворення в масив
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255  # Масштабування
    return img_array

# Прогнозування на новому зображенні
def predict_number(image_path):
    img_array = load_and_prepare_image(image_path)
    predictions = model.predict(img_array)  # Отримуємо прогнози
    predicted_class = np.argmax(predictions)  # Отримуємо клас з найбільшим значенням
    return predicted_class

# Тестування на новому зображенні
image_path = 'photo5.jpg'  # Вкажіть шлях до вашого зображення
predicted_number = predict_number(image_path)
print(f"The predicted number is: {predicted_number}")

input("Press Enter to exit...")