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

Let's step back a moment and think about the broad problem.
Suppose we have some outcome $y$ that we want to predict using an array of features $\boldsymbol{x}$.

$$
y = f(\boldsymbol{X}) + \varepsilon
$$

To make things a bit more concrete, let's assume we're doing linear regression.
The equation we're estimating
$$
\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

Or, looking at a single prediction

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \varepsilon
$$


Turns out we have an analytical solution for \boldsymbol{\hat{\beta}} the best estimator of \boldsymbol{\beta}:

$$
\hat{\boldsymbol{\beta}} = \left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}
$$

But what is $X$ here? What if it has categorical data?
Typically matrix multiplication has numeric values.
We'll next see two ways of making the transformation from categorical to numeric.

```{python}
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()
```

Two approaches:

1. Factorization: assign each original value a unique integer.

```python
codes, labels = pd.factorize(tips['day'])
codes
```

Let's try that

```python
columns = ['sex', 'smoker', 'day', 'time']
tips_factorized = tips.copy()

tips_factorized[columns] = tips[columns].apply(lambda x: pd.factorize(x)[0])
tips_factorized.head()
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
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.pipeline import TransformerMixin

class CategoricalTransformer(TransformerMixin):
```

Typically your custom transformers will inherit from `TransformerMixin`.
This gives us the `fit_transform` method, and also signals the intended use
of the class.


```python
    def __init__(self, drop_first=False):
        """
        Dummy encode `Categorical` dtype columns.

        Parameters
        ----------
        drop_first : bool
            whether to drop the first category per column.

        See Also
        --------
        pandas.get_dummies
        """
        self.drop_first = drop_first
```

The `fit` method ideally just takes `X` and `y`, so we pass and store all
custom parameters we'll need when the instance is created.

The basic goal of `fit` is to transform from Categorical to dummy-encoded, *while tracking how to get back to the original*. We have a few items to keep track of

- Which columns are categorical?
- What are the actual categories of the categorical columns?
- Which categoricals are ordered?
- What new column names do the original categories become?

```
    def fit(self, X, y=None, *args, **kwargs):
        """
        Parameters
        ----------
        X : DataFrame
        y : array-like, optional

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat.categories
                         for col in self.cat_columns_}
        self.ordered_ = {col: X[col].cat.ordered
                         for col in self.cat_columns_}

        n = 1 if self.drop_first else 0
        self.dummy_columns_ = {col: ["_".join([col, str(v)])
                               for v in self.cat_map_[col][n:]]
                               for col in self.cat_columns_}

        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )
        return self
```

So

The transform method is relatively straightforward.

```python
    def transform(self, X, y=None, *args, **kwargs):
        return (pd.get_dummies(X, drop_first=self.drop_first)
                  .reindex(columns=self.transformed_columns_)
                  .fillna(0))
```

An `inverse_transform` from dummy-encoded to categorical is nice to have.

```python
    def inverse_transform(self, X):
        X = np.asarray(X)
        non_cat_cols = (self.transformed_columns_
                            .get_indexer(self.non_cat_columns_))
        non_cat = pd.DataFrame(X[:, non_cat_cols],
                               columns=self.non_cat_columns_)

        series = []
        for col, cat_cols in self.dummy_columns_.items():
            locs = self.transformed_columns_.get_indexer(cat_cols)
            codes = X[:, locs].argmax(1)
            if self.drop_first:
                codes += 1
                codes[(X[:, locs] == 0).all(1)] = 0
            cats = pd.Categorical.from_codes(codes, self.cat_map_[col],
                                             ordered=self.ordered_[col])
            series.append(pd.Series(cats, name=col))
        # concats sorts, we want the original order
        df = (pd.concat([non_cat] + series, axis=1)
                .reindex(columns=self.columns_))
        return df
```
