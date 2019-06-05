# DATA 558: Polished Code Assignment

Frank Chen

## Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) installed, since we'll be using some sci-kit libraries.

## Setup

There are four Python scripts:

* `custom_logistic.py` - this contains the implementation of my custom LogisticRegression function
* `demo_real.py` - this contains the demo of my function on a real dataset ([spam](https://archive.ics.uci.edu/ml/datasets/spambase))
* `demo_sim.py` - this contains the demo of my function on a simulated dataset
* `exp_comparison.py` - this contains the accuracy comparison of my function vs. sklearn's LogisticRegression function

## Running the scripts

To run each script, simply execute it in the following manner:

```python
python3 <insert_script_name>
```

`exp_comparison.py` will ask for you to specify whether you'd like to see the comparison on the simulated or real dataset; the remaining scripts will not have any user input.

## Details of functions in `custom_logistics.py`

### `LogisticRegression` class

This class has 4 methods:

* `__init__(lam=1.0, max_iter=1000, print_iter=100, e=1e-3, log=True)` - initializes the class with the following variables:
    - lam - (default 1.0). Specifies the lambda value used for regularization.
    - max_iter - (default 1000). Specifies the  max number of iterations used in `coorddescent` or `fastgrad`
    - print_iter - (default 100). Specifies how many iterations in between each debug print statement
    - e - (default 1e-3). Specifies the error threshold for descent
    - log - (default True). Specifies whether to log info messages to the console during `fastgrad` algorithm or not.
* `fit(X, y)` - runs `fastgrad` algorithm to train coefficients to fit data `X` given labels `y`
* `visualize()` - creates the cost vs. iteration plot. This only makes sense after `fit` is run on the data.
* `score(X, y)` - creates an accuracy score of data `X` given correct labels `y`

Additionally, I have implemented the following functions that `LogisticRegression` class uses (details about each function is found in `custom_logistic.py`)

* fastgradalgo
* backtracking
* computegrad
* computecost