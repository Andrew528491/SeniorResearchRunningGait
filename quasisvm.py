'''Initial Results ML Test'''

from keras import models
from keras import layers
from keras.layers.experimental import RandomFourierFeatures
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

fileName = "extracted_videos.csv"
raw_data = open(fileName, "rt")
refined = np.loadtxt(raw_data, delimiter=",", dtype=str)
count = 0

y_values = refined[:, 0].reshape(len(refined), 1)
x_values = refined[:, 1:].astype(float)

y_values[y_values == "Good"]=1
y_values[y_values == "Bad"] = 0

y_values=y_values.astype(float)

train_x = x_values[0:500, :]
test_x = x_values[500:, :]

train_y = y_values[0:500, :].astype(float)
test_y = y_values[500:, :].astype(float)

#create the model
model = models.Sequential(
    [
        models.Input(shape=(2277,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=1, activation='sigmoid'),
    ]
)
model.compile(
    optimizer='rmsprop',
    loss='hinge',
    metrics=['accuracy'],
)

'''
network = models.Sequential()
network.add(layers.Dense(200, activation='sigmoid', input_shape=(2970,)))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer = 'rmsprop', 
                loss='binary_crossentropy',
                metrics=['accuracy'])
'''

#train and evaluate the images
model.fit(train_x, train_y, epochs=8, batch_size=40)
test_loss, test_acc = model.evaluate(test_x, test_y)
print('test_acc:', test_acc)
print (model.summary())

predicted = model.predict(x_values)

predicted[predicted >= 0.5] = 1
predicted[predicted < 0.5] = 0

true_positive = np.sum([predicted[y_values==1]==1])
true_negative = np.sum([predicted[y_values==0]==0])

false_positive = np.sum([predicted[y_values==0]==1])
false_negative = np.sum([predicted[y_values==1]==0])

confusion_matrix = [["n = 10000", "Predicted Bad", "Predicted Good"], 
                    ["Actual Bad", str(true_negative), str(false_positive)], 
                    ["Actual Good", str(false_negative), str(true_positive)]]

print(tabulate(confusion_matrix))

accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
error_rate = 1-accuracy

# Important when cost of a false positive is high
precision = (true_positive)/(true_positive+false_positive)

# Important when cost of a false negative is high
recall = (true_positive)/(true_positive+false_negative)

model_reliability = [["Accuracy:", str(round(accuracy, 3))], 
                     ["Error Rate:", str(round(error_rate, 3))],
                     ["Precision:", str(round(precision, 3))],
                     ["Recall:", str(round(recall, 3))]]

# Print formatted data table of measures of model reliability
print(tabulate(model_reliability))


