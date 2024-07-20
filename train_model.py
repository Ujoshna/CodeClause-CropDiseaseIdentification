import json

import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


def load_data(dataset):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)
    train_generator = datagen.flow_from_directory(dataset,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset='training'
                                                  )
    validation_generator = datagen.flow_from_directory(
        dataset,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator


dataset_path = "PlantVillage/"
train_generator, validation_generator = load_data(dataset_path)
# Save class indices
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)


# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save('model.h5')
# model.save('my_model.keras')
