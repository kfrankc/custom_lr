import custom_logistic as cl
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use demo spam dataset
spam = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', sep=' ', header=0)
X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]
y = y.replace(0,-1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Divide the data into training and test sets. By default, 25% goes into the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Convert to np array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Create trained LR classifer object on training data
clf = cl.LogisticRegression(lam=0.1, e=1e-4, print_iter=20).fit(X_train, y_train)

clf.visualize()

print("Performance (Accuracy) on Training Data: {}".format(clf.score(X_train, y_train)))
print("Performance (Accuracy) on Test Data: {}".format(clf.score(X_test, y_test)))