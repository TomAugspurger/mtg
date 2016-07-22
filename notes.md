---
title: "Bridging the pandas -- scikit-learn dtype divide"
author: "Tom Augspurger"
date: "2016-07-22"
output:
  pdf_document: default
  word_document: default
  html_document:
    keep_md: yes
---

Notes for my talk at PyData Chicago 2016

# The Problem

Pandas and scikit-learn have overlapping, but different data models.
Both are based off NumPy arrays, but the extensions pandas has made to NumPy's type system have created a rift between the two. Most notably

- Homogeneity
  + NumPy arrays (and so scikit-learn feature matrices) are *homogeneous*, they must have a single dtype.
  + Pandas DataFrames, can store *heterogeneous* data

# Statistical Side

Stepping back, let's thing about our goal.
Suppose we have some data `X` (a NumPy array or DataFrame) and we want to predict `y`, a vector.
To keep things extremely simple, we'll only focus on the linear regression case (OLS).

```{python}
X = pd.DataFrame({'A': ['a', 'a', 'b', 'a', 'b', 'b'],
                  'B': [1, 2, 3, 3, 2, 4]})
y = pd.Series([3, 4, 7, 5, 5, 8])
```

The equation we're estimating

$$
\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
$$


Turns out

$$
\hat{\boldsymbol{\beta}} = \left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}
$$

But what is $X$ here? Typically our matrix multiplication has numeric values.
Two approaches.

1. Factorization: assign each original value a unique integer.

```{python}
codes, labels = pd.factorize(X['A'])
codes
```

Let's try that

```{python}
X_factoriezed = X.copy()
X['A'] = codes
```


2. Dummy-encoding

This fixes some of the problems above.

```{python}
dummies = pd.get_dummies(X)
dummies
```


# NumPy

![dtype Hierarchy](figures/dtype-hierarchy.png)

In practice this isn't such a big deal.
In order to fit the model, we'll *usually* [^1: categorical trees?] need to convert everything to a numeric matrix.
However, you don't want to irreparably lose the richer information of a categorical.

# `CategoricalTransformer`


We implement a `CategoricalTransformer`.

```{python}
import numpy as np
import pandas as pd
from sklearn.pipeline import TransformerMixin


class CategoricalTransformer(TransformerMixin):

    def fit(self, X, y=None, *args, **kwargs):
        self.columns_ = X.columns
        self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat.categories
                         for col in self.cat_columns_}
        self.ordered_ = {col: X[col].cat.ordered
                         for col in self.cat_columns_}

        self.dummy_columns_ = {col: ["_".join([col, v])
                                     for v in self.cat_map_[col]]
                               for col in self.cat_columns_}
        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )

    def transform(self, X, y=None, *args, **kwargs):
        return (pd.get_dummies(X)
                  .reindex(columns=self.transformed_columns_)
                  .fillna(0))

    def inverse_transform(self, X):
        X = np.asarray(X)
        series = []
        non_cat_cols = (self.transformed_columns_
                            .get_indexer(self.non_cat_columns_))
        non_cat = pd.DataFrame(X[:, non_cat_cols],
                               columns=self.non_cat_columns_)
        for col, cat_cols in self.dummy_columns_.items():
            locs = self.transformed_columns_.get_indexer(cat_cols)
            codes = X[:, locs].argmax(1)
            cats = pd.Categorical.from_codes(codes, self.cat_map_[col],
                                             ordered=self.ordered_[col])
            series.append(pd.Series(cats, name=col))
        # concats sorts, we want the original order
        df = (pd.concat([non_cat] + series, axis=1)
                .reindex(columns=self.columns_))
        return df
```



