from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

#get the imdb dataset
#traindata is a list of reviews
#trainlabels list of 0 and 1: 1 is positive 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#preparing the dataset, turning LISTS into tensors
def vectorize_sequences(sequences, dimension=10000): #
    results = np.zeros((len(sequences), dimension)) #creates an all zero matrix of shape
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. #sets specific indices of results
    return results

x_train = vectorize_sequences(train_data) #vectors
x_test = vectorize_sequences(test_data)

#vectorize LABELS
y_train = np.asarray(train_labels).astype('float32') # converts input into an array /
y_test = np.asarray(test_labels).astype('float32')
#data is ready to be fed into neural network Dense layers with relu activation

#setting up models and layers
model = models.Sequential() #Sequential groups a linear stack of layers into a model
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#compiling model
#loss function: crossentropy measures distance between probability distribution
#optimizer: rmsprop
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#configure optimizer
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#using custom losses and metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

#validating the approach - creating a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#training the model, 20 iternationa and mini batches of 512 samples, monitoring of loss and accurarcy 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

#plotting the training and validation loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

"""
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
"""
print(model.predict(x_test))
