# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing convolutional neural network
classifier = Sequential()

#step-1 first convolution layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step-2 Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step-3 convolution  layer
classifier.add(Convolution2D(32,3,3,activation='relu'))

#step-4 Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step-5 Flattening
classifier.add(Flatten())

#step-4 Fully connection
classifier.add(Dense(output_dim= 128,activation='relu'))
classifier.add(Dense(output_dim= 1,activation='sigmoid'))

#Cimpiling th CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#get_ipython().system_raw("unrar x training_set.rar")

#get_ipython().system_raw("unrar x test_set.rar")

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')

# Create a loss history
history = LossHistory()

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         max_q_size=100,
                         callbacks=[history])

# Save model
model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)

# Save loss history to file
loss_history_path = os.path.join(script_dir, '../loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()

backend.clear_session()
print("The model class indices are:", training_set.class_indices)
