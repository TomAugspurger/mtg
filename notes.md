---
title: "Bridging the pandas -- scikit-learn dtype divide"
author: "Tom Augspurger"
date: "2016-07-22"
---

Notes for my talk at PyData Chicago 2016

# The Problem

Pandas and scikit-learn have overlapping, but different data models.
Both are based off NumPy arrays, but the extensions pandas has made to NumPy's type system have created a rift between the two. Most notably

1. Homogeneity
    - NumPy arrays (and so scikit-learn feature matrices) are *homogeneous*, they must have a single dtype.
    - Pandas DataFrames, can store *heterogeneous* data
2. Extension Types
    - Pandas has implemented several extension dtypes, including `Categorical` and Datetime with TZ.

"Real-world" data is often heterogeneous, making pandas the tool of choice.
However, tools like Scikit-Learn, which do not depend on pandas, can't use its
richer data structures.
We need a way of bridging the gap between pandas' DataFrames and NumPy the arrays appropriate for scikit-learn.

# Statistical Side

Suppose we want to predict `tip` using all the other variables.
To keep things extremely simple, we'll only focus on the linear regression case (OLS), though this approach is useful more generally.

```{python}
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()
```

The equation we're estimating

$$
\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$


Turns out

$$
\hat{\boldsymbol{\beta}} = \left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}
$$

But what is $X$ here? Typically matrix multiplication has numeric values.
We need to convert the categorical data to numeric.
Two approaches.

1. Factorization: assign each original value a unique integer.

```python
codes, labels = pd.factorize(tips['day'])
codes
```

Let's try that

```python
columns = ['sex', 'smoker', 'day', 'time']
X_factorized = tips.copy()

X_factorized[columns] = tips[columns].apply(lambda x: pd.factorize(x)[0])
X_factorized.head()
```

However, there are several problems with this approach.
First, ordering becomes important.
If 'b' happened to come before 'a' on next time around, you're results would change.

Second, it asserts that the difference between any two "adjacent" categories is the same.
That is, the change in $y$ with respect to a jump from `'Thusrday'` to `'Friday'` has the same effect as a jump from `'Friday'` to `'Saturday'`.
Sometimes this may be true, but we have no reason to believe that here.

2. Dummy-encoding

This fixes the problems above, and is the approach we'll use.

```python
tips_dummies = pd.get_dummies(tips, drop_first=True)
tips_dummies.head()
```

We can now fit the regression

```python
from sklearn.linear_model import LinearRegression

X, y = tips_dummies.drop("tip", axis=1), tips_dummies["tip"]
lm = LinearRegression().fit(X, y)

yhat = lm.predict(X)
plt.scatter(y, y - yhat)
```

## `CategoricalTransformer`

`Pipeline`s are great.
Use them wherever possible.
Convenient for grid searching.
Quarantines test data from training.

Our example above has a couple issues if we were going to "productionize" it.

1. A bit difficult to go from dummy-encoded back to regular. Pandas doesn't have a `from_dummies` yet (PR anyone?)
2. If working with a larger dataset and `partial_fit`, codes could be missing from subsets of the data. This *should* be OK, as long as you're careful to construct
the DataFrame with all the categoricals. It'd be nice to have a sanity check though.

We implement a `CategoricalTransformer`.

```python
code
```



