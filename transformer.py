from itertools import chain

import numpy as np
import pandas as pd
from sklearn.pipeline import TransformerMixin


class CategoricalTransformer(TransformerMixin):

    def __init__(self, drop_first=False):
        # TODO
        self.drop_first = drop_first

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

        self.dummy_columns_ = {col: ["_".join([col, v])
                                     for v in self.cat_map_[col]]
                               for col in self.cat_columns_}
        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )
        return self

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
