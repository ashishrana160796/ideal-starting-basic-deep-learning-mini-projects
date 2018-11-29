# Sentiment Analysis of tweets in context to GOP debates in 2016 
# Link used as reference: https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras/notebook

# import statements

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# reading data from csv dataset
data = pd.read_csv('dataset/Sentiment.csv')
# keeping only the text and sentiment part
data = data[['text','sentiment']]

# dropping neutral data, only positive and negative data considered
data = data[data.sentiment != "Neutral"] 

# max features limit to 2000, cleaning with regex and tokenize to form a sequence for feeding into network.

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# Output:
# 4472
# 16986


# embed_dim, lstm_out, batch_size, droupout_x as hyperparameters for tuning.
# categorical crossentropy, adam optimizer and softmax used in network.
embed_dim = 128
lstm_out = 196


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# Output: 

# (7188, 28) (7188, 2)
# (3541, 28) (3541, 2)


model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, 28, 128)           256000    
# _________________________________________________________________
# spatial_dropout1d_1 (Spatial (None, 28, 128)           0         
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 196)               254800    
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 394       
# =================================================================
# Total params: 511,194
# Trainable params: 511,194
# Non-trainable params: 0
# _________________________________________________________________
# None

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

# Output:

# Epoch 1/7
# 2018-11-28 23:11:22.297329: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# - 16s - loss: 0.4397 - acc: 0.8173
# Epoch 2/7
# - 16s - loss: 0.3231 - acc: 0.8653
# Epoch 3/7
# - 19s - loss: 0.2856 - acc: 0.8790
# Epoch 4/7
# - 19s - loss: 0.2621 - acc: 0.8925
# Epoch 5/7
# - 18s - loss: 0.2348 - acc: 0.9015
# Epoch 6/7
# - 18s - loss: 0.2124 - acc: 0.9150
# Epoch 7/7
# - 19s - loss: 0.1900 - acc: 0.9225
# Out[7]: <keras.callbacks.History at 0x7f789b9a3e10>


# Save the model for further usage in future
model.save('sentiment-analyzer.h5')
# for loading the model
# model = load_model('sentiment-analyzer.h5')


# Validation for a sample batch
validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

# Validation Results:
# score: 0.46
# acc: 0.83


# Checking accuracy for both positive and negative classes i.e. validation for tweets for both classes.
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

# Results for positive and negative reviews:
# pos_acc 60.517799352750814 %
# neg_acc 91.43576826196474 %
# Inference: Bad Results for positive values. Probable cause: Unbalanced dataset.


# Sample tweet for testing our result
twt = ['Meetings: Because none of us is as dumb as all of us.']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")

# Result is as follow:    
# [[   0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0  206  633    6  150    5   55 1055   55   46    6  150]]
# negative
