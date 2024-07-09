import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

abalone_train = pd.read_csv('abalone.csv',sep=",",
                            names=["Length", "Diameter", "Height", "Whole weight",
"Shucked weight", "Viscera weight", "Shell weight", "Age"])

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")

abalone_features = abalone_features.astype('float32')
abalone_labels = abalone_labels.astype('float32')

model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(1)
])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

fitting = model.fit(abalone_features, abalone_labels, epochs=10, verbose=0)

print(fitting.history)
model.summary()