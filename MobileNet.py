# Needed libraries
import sys
import os
import datetime
import keras as keras
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

from keras import backend as k
from shutil import copyfile
import functools

# Fix seed for reproducible results (only works on CPU, not GPU)
seed = 5050
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# Hyper parameters for model
train_size = 26752
val_size = 3855
nb_classes = 257  # number of classes
img_width, img_height = 224, 224  # change based on the shape/structure of your images
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 100  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # learning rate
decay_rate = 1e-6 # lr decay rate
based_model_last_block_layer_number = 126  # value is based on based model selected.

# Load Xception
#base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
base_model = MobileNet(input_shape=(img_width, img_height, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None)
for layer in base_model.layers:
        layer.trainable = False # don't train Xception layers
#print(base_model.summary())

# Top Model Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# Add your top layer block to your base model
model = Model(base_model.input, predictions)
print(model.summary())

#Paths
now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
train_data_dir = '/home/refaay/Downloads/Work/assignment3/Train'
validation_data_dir = '/home/refaay/Downloads/Work/assignment3/Validate'
save_dir = '/home/refaay/Downloads/Work/assignment3/augmented/' + now
model_path = '/home/refaay/Downloads/Work/assignment3/modelweights/'+ now
code_path = '/home/refaay/Downloads/Work/assignment3/codes/' + now
log_path = '/home/refaay/Downloads/Work/assignment3/logs/' + now
top_weights_path = model_path + '/top_model_weights.h5'
final_weights_path = model_path + '/model_weights.h5'
#top_weights_path = model_path + '/' + weights.{epoch:02d}-{val_acc:.5f}.hdf5

os.makedirs(code_path)
os.makedirs(save_dir)
os.makedirs(model_path)
os.makedirs(log_path)
copyfile('/home/refaay/Downloads/Work/assignment3/assignment3.py', code_path+'/assignment3.py')

# Training data generation
train_datagen = ImageDataGenerator(rescale=1. / 255, #samplewise_center=True,  samplewise_std_normalization=True, # 0 mean & normalization
				    shear_range=transformation_ratio, zoom_range=transformation_ratio, # zca_whitening=True, rotation_range=1, 
                                    cval=transformation_ratio, horizontal_flip=True, vertical_flip=True) # vertical_flip=False)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical',
                                                    save_to_dir=save_dir, save_prefix='aug', save_format='jpeg')

# Validation data generation
validation_datagen = ImageDataGenerator(rescale=1. / 255) #ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True) # 0 mean & normalization

validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                                  batch_size=batch_size, class_mode='categorical')

top5 = functools.partial(keras.metrics.top_k_categorical_accuracy, k = 5)
top5.__name__ = "top5_acc"

# Callbacks
checkpointer = ModelCheckpoint(top_weights_path, monitor='val_top5_acc', verbose=1, save_best_only=True, save_weights_only=False) # saves model
tensorboard = keras.callbacks.TensorBoard(log_dir= log_path, histogram_freq=0, write_graph=True, write_images=False) # graphs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6) # reduces learning rate
stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0) # stops if no improvement
callbacks_list = [checkpointer, tensorboard, reduce_lr] # last terminates on NAN loss values -> , stopping, TerminateOnNaN()

# Model compilation
adam = keras.optimizers.Adam(lr=learn_rate, decay=decay_rate) # Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', top5])

# Train Simple CNN
model.fit_generator(train_generator, steps_per_epoch=train_size/batch_size,
                    epochs=nb_epoch / 5, verbose = 1,
                    validation_data=validation_generator, validation_steps=val_size/batch_size,
                    callbacks=callbacks_list)

# Fine tuning
print("\nStarting to Fine Tune Model\n")
model.load_weights(top_weights_path)  # add the best weights from the train top model

# Train last only
for layer in model.layers[:based_model_last_block_layer_number]:
	layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
	layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', top5])
callbacks_list = [checkpointer, tensorboard, reduce_lr] # last terminates on NAN loss values -> , stopping, TerminateOnNaN()
model.fit_generator(train_generator, steps_per_epoch=train_size/batch_size,
                    epochs=nb_epoch, verbose = 1,
                    validation_data=validation_generator, validation_steps=val_size/batch_size,
                    callbacks=callbacks_list)

# Save model
model_json = model.to_json()
with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
	json_file.write(model_json)

# Release memory
k.clear_session()