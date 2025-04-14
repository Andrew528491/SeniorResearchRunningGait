import numpy as np
from tabulate import tabulate
from sklearn import svm

fileName = "extracted_videos.csv"
raw_data = open(fileName, "rt")
refined = np.loadtxt(raw_data, delimiter=",", dtype=str)
count = 0

y_values = refined[:, 0]
x_values = refined[:, 1:].astype(float)

y_values[y_values == "Good"]=1
y_values[y_values == "Bad"] = 0

y_values=y_values.astype(float)

#linear_svm = svm.SVC(kernel='linear')
linear_svm = svm.SVC()

train_x = x_values[0:400, :]
test_x = x_values[400:, :]

train_y = y_values[0:400].astype(float)
test_y = y_values[400:].astype(float)

linear_svm.fit(train_x, train_y)

predicted = linear_svm.predict(test_x)

true_positive = np.sum([predicted[test_y==1]==1])
true_negative = np.sum([predicted[test_y==0]==0])

false_positive = np.sum([predicted[test_y==0]==1])
false_negative = np.sum([predicted[test_y==1]==0])

confusion_matrix = [["", "Predicted Bad", "Predicted Good"], 
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