import numpy as np
import tensorflow as tf

# Cargar datos de entrenamiento
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocesamiento de los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Crear modelo de PNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((784,), input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluar modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)