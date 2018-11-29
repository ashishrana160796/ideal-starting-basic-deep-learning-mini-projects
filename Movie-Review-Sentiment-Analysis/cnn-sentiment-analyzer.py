# CNN for the IMDB problem
# Concept applied here as the techniques invariance to the specific position of features is still valid for word in paragraph.
# And their association to particular qualities while classifying reviews much similar to pixels associated with neigbourhood.
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
# Conv1D layer with 32 feature map along with 1D max pooling layer of stride 2.
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Output:

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, 500, 32)           160000    
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 500, 32)           3104      
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 8000)              0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 250)               2000250   
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 251       
# =================================================================
# Total params: 2,163,605
# Trainable params: 2,163,605
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 25000 samples, validate on 25000 samples


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Output: 

# - 25s - loss: 0.4885 - acc: 0.7345 - val_loss: 0.2828 - val_acc: 0.8838
# Epoch 2/2
# - 25s - loss: 0.2262 - acc: 0.9105 - val_loss: 0.2739 - val_acc: 0.8866
# Accuracy: 88.66%
