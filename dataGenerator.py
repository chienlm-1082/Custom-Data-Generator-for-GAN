import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import load_img
import os
from tqdm import tqdm

train_path = os.listdir('gen/')

# load data mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
class DataGenerator(Sequence):
    def __init__(self,
                 img_paths,
                 labels, 
                 batch_size=32,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=4,
                 shuffle=True,
                 src_folder='',
                 tar_folder=''):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.img_paths = img_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.src_path = src_folder
        self.tar_path = tar_folder
        self.img_indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *self.dim))
        y = []
        for i, ID in enumerate(list_IDs_temps):
            # X[i,] = self.img_paths[ID]
            X[i,] = load_img(self.src_path + self.img_paths[ID], target_size=self.dim)
            Y[i,] = load_img(self.tar_path+self.img_paths[ID], target_size=self.dim)
            X = (X/255).astype('float32')
            Y = (Y/255).astype('float32')
        return X, Y
input_shape = (512, 512, 3)
n_classes = 10
train_generator = DataGenerator(train_path, y_train, batch_size = 2, dim = input_shape,
 n_classes=10, shuffle=True, src_folder='gen/', tar_folder='target/')

print(train_generator.__getitem__(0)[0].shape)
print(len(train_generator))

for i in tqdm(range(len(train_generator))):
  src = train_generator[i][0]
  tar = train_generator[i][1]
  print(src.shape, tar.shape)