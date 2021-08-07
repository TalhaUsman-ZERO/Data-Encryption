#importing libraries
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#tf.config.run_functions_eagerly(True)

#importing the dataset
dataset = pd.read_csv("preserved.csv", delimiter=",")
X = dataset.iloc[:,1:5].values
labelencoder=LabelEncoder()
y = labelencoder.fit_transform(dataset.iloc[:, 5].values)


#splitting the dataset
X_train, X_test, y_train , y_test=train_test_split(X,y,test_size=0.20, random_state=0)
X_train, X_validate, y_train , y_validate=train_test_split(X_train,y_train,test_size=0.20, random_state=0)  

#feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_validate=sc.fit_transform(X_validate)
X_test=sc.transform(X_test)

# create model layers and Model

inputs = keras.Input(shape=(4,))

dense = layers.Dense(12, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
dense = layers.Dropout(0.25)(dense)
dense = layers.Dense(12, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001))(dense)
dense = layers.Dropout(0.25)(dense)
dense = layers.Dense(12, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001))(dense)
dense = layers.Dropout(0.25)(dense)

outputs = layers.Dense(3, activation="softmax")(dense)

model = keras.Model(inputs=inputs, outputs=outputs, name="Sequential_Model")

model.summary()


#adding the input layer and first hidden layer
#model.add(Dense(20, input_dim=4, activation='elu'))
#model.add(Dropout(0.25))

#adding the second hidden layer
#model.add(Dense(20, activation='elu'))
#model.add(Dropout(0.25))

#adding the output layer
#model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #optimizer='rmsprop'

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint.best = 0.7
callbacks_list = [checkpoint]


y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)
y_validate = keras.utils.to_categorical(y_validate, 3)

#fitting the model
history = model.fit(X_train, y_train,
          validation_data=(X_validate,y_validate),
          epochs=100,
          batch_size=10,
          callbacks=callbacks_list,
          verbose=1)

test_acc=model.evaluate(X_test,y_test)

print(test_acc)















