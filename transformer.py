from itertools import chain

import numpy as np
import pandas as pd
from sklearn.pipeline import TransformerMixin


class CategoricalTransformer(TransformerMixin):

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
        self.dummy_columns_ = {col: ["{}_{}".format(col, v)
                               for v in self.cat_map_[col][n:]]
                               for col in self.cat_columns_}

        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )
        return self

    def transform(self, X, y=None, *args, **kwargs):
        return (pd.get_dummies(X, drop_first=self.drop_first)
                  .reindex(columns=self.transformed_columns_)
                  .fillna(0))

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
