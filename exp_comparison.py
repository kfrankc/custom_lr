import custom_logistic as cl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("My LR classifier vs. sklearn.linear_model.LogisticRegression")
print("------------------------------------------------------------")
option = int(input("Press [0] for simulated data; [1] for spam data: "))
if (option == 0):
    # Create a simulated feature matrix and output vector with 100 samples,
    X, y = make_classification(n_samples = 100, n_features = 10, \
                            n_classes = 2, weights = [.4, .6])
    X = np.array(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(y).reshape(-1, 1)
    y[y==0] = -1

    # Create trained LR classifer object
    clf = cl.LogisticRegression(log=False).fit(X, y)
    print("My LR classifer Performance (Accuracy): {}".format(clf.score(X, y)))

    # Create trained sklearn LogisticRegression
    skclf = LogisticRegression(solver='lbfgs').fit(X, y.ravel())
    print("Sklearn LogisticRegression Performance (Accuracy): {}".format(skclf.score(X, y)))
elif (option == 1): 
    spam = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', sep=' ', header=0)
    X = spam.iloc[:,:-1]
    y = spam.iloc[:,-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.replace(0,-1)
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
    clf = cl.LogisticRegression(lam=0.1, e=1e-4, log=False).fit(X_train, y_train)
    print("My LR classifier Performance (Accuracy) on Training Data: {}".format(clf.score(X_train, y_train)))
    print("My LR classifier Performance (Accuracy) on Test Data: {}".format(clf.score(X_test, y_test)))

    # Create sklearn LogisticRegression on training data
    skclf = LogisticRegression(solver='lbfgs').fit(X_train, y_train.ravel())
    print("Sklearn LogisticRegression Performance (Accuracy) on Training Data: {}".format(skclf.score(X_train, y_train)))
    print("Sklearn LogisticRegression Performance (Accuracy) on Test Data: {}".format(skclf.score(X_test, y_test)))
else:
    raise("Please choose 0 or 1 only")