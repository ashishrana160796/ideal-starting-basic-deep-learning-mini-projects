# Main Script Starts Here

# import statements listed

# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# summarize size
# print("Training data: ")
# print(X.shape)
# print(y.shape)

# Output:

# Training data: 
# (50000,)
# (50000,)


# Summarize number of classes
# print("Classes: ")
# print(numpy.unique(y))

# Output:

# Classes: 
# [0 1]


# Summarize number of words
# print("Number of words: ")
# print(len(numpy.unique(numpy.hstack(X))))

# Output:

# Number of words: 
# 88585


# Summarize review length
# print("Review length: ")
# result = [len(x) for x in X]
# print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length
# pyplot.boxplot(result)
# pyplot.show()

#Output:

# Matplot Graph Review Available

# Review length: 
# Mean 234.76 words (172.911495)


# Word Embeddings Provided By Keras is Being Used here.
# imdb.load_data(nb_words=5000)

# Output:
# Array of Vectors stored as list

# Padding the Text
# from keras.preprocessing.sequence import sequence                      
# X_train = sequence.pad_sequences(X_train, maxlen=500) 
# X_test = sequence.pad_sequences(X_test, maxlen=500)                 


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Padding The Sequence
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, 500, 32)           160000    
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 16000)             0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 250)               4000250   
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 251       
# =================================================================
# Total params: 4,160,501
# Trainable params: 4,160,501
# Non-trainable params: 0
# _________________________________________________________________
# None


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Results:
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# - 23s - loss: 0.5086 - acc: 0.7130 - val_loss: 0.3334 - val_acc: 0.8528
# Epoch 2/2
# - 23s - loss: 0.1907 - acc: 0.9274 - val_loss: 0.3010 - val_acc: 0.8729
# Accuracy: 87.29%

