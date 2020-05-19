import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

    train_data_dir = 'G:/natural/classes/train/'
    validation_data_dir = 'G:/natural/classes/valid/'

    model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(500,500, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(momentum=0.9),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
 
    tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,  
        target_size=(500,500), 
        batch_size=20,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir, 
        target_size=(500,500), 
        batch_size=5,
        class_mode='categorical')

    history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
    model.save('model.h5')
 
    
