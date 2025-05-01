import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
 
 
input1 = Input(shape=(10,))
x1 = Dense(64, activation='relu')(input1)
x1 = Dense(32, activation='relu')(x1)
 
 
input2 = Input(shape=(20,))
x2 = Dense(64, activation='relu')(input2)
x2 = Dense(32, activation='relu')(x2)
 
# Concatenate the outputs of the two branches
concatenated = Concatenate()([x1, x2])
 
# Add a final output layer
output = Dense(1, activation='sigmoid')(concatenated)
 
 
model = Model(inputs=[input1, input2], outputs=output)
 
model.summary()