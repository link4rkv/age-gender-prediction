import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings

import cv2 as cv

import matplotlib.pyplot as plt

img_path = "../input/utkface-new/UTKFace"

img_files = os.listdir(img_path)

SAMPLE_SIZE = 10000
IMAGE_SIZE = 128

labels = []
images = []

i = 0
while(i < SAMPLE_SIZE):
    labels.append([[int(img_files[i].split('_')[0])], [int(img_files[i].split('_')[1])]])
    
    img = cv.imread(img_path + '/' + img_files[i])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    images.append(img)
    
    i += 1

X = np.array(images) / 255
Y = np.array(labels)

print(X[0:2])
print(Y[0:2])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)

Y_train_final = [Y_train[:, 0], Y_train[:, 1]]
Y_test_final = [Y_test[:, 0], Y_test[:, 1]]

values, counts = np.unique(Y[:, 0], return_counts = True)

fig = plt.figure()
axes = fig.add_axes([0, 0, 2, 2])
axes.bar(values, counts)
plt.show()

values, counts = np.unique(Y[:, 1], return_counts=True)

fig = plt.figure()
axes = fig.add_axes([0, 0, 1, 1])
axes.bar(['Male', 'Female'], [counts[0], counts[1]])
plt.show()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

initial_model = keras.Sequential([
    layers.Conv2D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same',
                  input_shape = [IMAGE_SIZE, IMAGE_SIZE, 3]),
    layers.MaxPool2D(),

    layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(),

    layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),

    layers.Dense(1, activation = 'relu', name = 'age'),
    layers.Dense(1, activation = 'sigmoid', name = 'gender'),
])

model = keras.Model(
     inputs = initial_model.inputs,
     outputs = [initial_model.get_layer(name = "age").output, initial_model.get_layer(name = "gender").output]
)

model.compile(
    optimizer = 'adam',
    loss = ['mse', 'binary_crossentropy'],
    metrics = ['accuracy']
)

history = model.fit(
    X_train,
    Y_train_final,
    batch_size = 32,
    validation_data = (X_test, Y_test_final),
    epochs = 20,
)

i = 3316
gender = ['Male', 'Female']
print("Actual Age: " + str(int(img_files[i].split('_')[0])))
print("Actual Gender: " + gender[int(img_files[i].split('_')[1])])

image = X[i]
prediction = model.predict(np.array([image]))
print("Predicted Age: " + str(int(np.round(prediction[0][0]))))
print("Predicted Gender: " + gender[int(np.round(prediction[1][0]))])


import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['gender_accuracy', 'val_gender_accuracy']].plot();

predictions = model.predict(X_test)

plt.scatter(Y_test_final[0], predictions[0], color = 'blue')
plt.plot([Y_test_final[0].min(), Y_test_final[0].max()], [Y_test_final[0].min(), Y_test_final[0].max()], color = 'black', linewidth = 3)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.show()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns

gender_pred = list(map(lambda e : int(np.round(e)), predictions[1]))

report = classification_report(Y_test_final[1], gender_pred)
results = confusion_matrix(Y_test_final[1], gender_pred)

print(report)
sns.heatmap(results, annot = True)
