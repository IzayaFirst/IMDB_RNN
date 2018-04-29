from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

max_features = 10000
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test  = sequence.pad_sequences(input_test,  maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test  shape:', input_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['acc'])

history = model.fit(input_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2)

acc      = history.history['acc']
val_acc  = history.history['val_acc']
loss     = history.history['loss']
val_loss = history.history['val_loss']

scores = model.evaluate(input_train, y_train)
print(scores)

scores = model.evaluate(input_test, y_test)

print(scores)
