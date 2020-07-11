import time
NAME = 'mask_detector_' + str(int(time.time()))
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras.regularizers import l2
train_datagen = ImageDataGenerator(rescale=1./255)

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

train_generator = train_datagen.flow_from_directory(
                    'Training/',
                    target_size=(100,100),
                    batch_size=128,
                    class_mode='binary'
)


test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
                        'Validation/',
                        target_size=(100,100),
                        batch_size=32,
                        class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                            input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, 
                        kernel_regularizer=l2(0.01),
                        bias_regularizer=l2(0.01),
                        activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.001),
                metrics=['acc']
)

history = model.fit_generator(
    train_generator,
    epochs=50,
    validation_data = validation_generator,
    verbose=2,
    callbacks=[tensorboard]
)

model.save(NAME)

print(model.summary())