import custom_logistic as cl
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create a simulated feature matrix and output vector with 100 samples,
X, y = make_classification(n_samples = 100, n_features = 10, \
                           n_classes = 2, weights = [.4, .6])

X = np.array(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y).reshape(-1, 1)
y[y==0] = -1

# Create trained LR classifer object
clf = cl.LogisticRegression().fit(X, y)

clf.visualize()

print("Performance (Accuracy): {}".format(clf.score(X, y)))
