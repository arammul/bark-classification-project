import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import applications
import zipfile
from keras.layers import Input
from tensorflow.keras.models import Model
# Specify the path to your zip file
zip_file_path = 'toy_data.zip'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall()


train = list(os.walk('toy_data')) # adjust depending on where you store the data

label_names = train[0][1]
dict_labels = dict(zip(label_names, list(range(len(label_names)))))


def dataset(path, num_images_per_class=20):
    images = []
    labels = []
    for folder in tqdm(os.listdir(path)):  #Read only the first 4 folders
        value_of_label = dict_labels[folder]  # dict_labels is the dictionary whose key:value pairs are classes:numbers representing them
        image_count = 0

        for file in sorted(os.listdir(os.path.join(path, folder))):  # Sort to ensure consistent ordering
            path_of_file = os.path.join(os.path.join(path, folder), file)

            image = cv2.imread(path_of_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            images.append(image)
            labels.append(value_of_label)

            image_count += 1

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    return images, labels
dataset("toy_data")


# Define paths to your data directories
train_dir = 'toy_data'
test_dir = 'toy_data'

# Create TensorFlow datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

image_size = (224, 224)
batch_size = 64
train_datagen = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.4,
                            zoom_range = 0.4,
                            horizontal_flip = True,
                            vertical_flip = True,
                            validation_split = 0.2)

train_ds = train_datagen.flow_from_directory('toy_data',
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'training',
                                      color_mode="rgb",)


val_ds = train_datagen.flow_from_directory('toy_data',
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation',
                                      color_mode="rgb")


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,10))

for i in range(4):
    image = next(train_ds)[0][0]
    image = np.squeeze(image)
    ax[i].imshow(image)
    ax[i].axis(False)

vgg_base = applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))#resonate50
vgg_base.trainable = False

inputs = Input(shape=(224, 224, 3))

x = vgg_base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation = 'sigmoid')(x)
vgg_model = Model(inputs, outputs)
vgg_model.summary


vgg_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss= keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics= [keras.metrics.CategoricalAccuracy()],
)

epochs = 35
history = vgg_model.fit(train_ds, epochs=epochs, validation_data=val_ds)

vgg_model.save('vgg.hdf5') #this saves the model with the weights


#path
saved_model_path = 'model/VGG_forestry_model'

#saving the model
tf.saved_model.save(vgg_model, saved_model_path)

#convert in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

#path2
tflite_model_path = 'model/VGG_forestry_model.tflite'

#saving TensorFlow Lite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model saved succesfully at :", tflite_model_path)