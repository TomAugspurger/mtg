import numpy as np
import pandas as pd
from sklearn.pipeline import TransformerMixin


class DummyEncoder(TransformerMixin):

    def fit(self, X, y=None):
        self.index_ = X.index
        self.columns_ = X.columns
        self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat for col in self.cat_columns_}

        left = len(self.non_cat_columns_)
        self.cat_blocks_ = {}
        for col in self.cat_columns_:
            right = left + len(X[col].cat.categories)
            self.cat_blocks_[col], left = slice(left, right), right
        return self

    def transform(self, X, y=None):
        assert X.columns == self.columns_
        return np.asarray(pd.get_dummies(X))

    def inverse_transform(self, X):
        non_cat = pd.DataFrame(X[:, :len(self.non_cat_columns_)],
                               columns=self.non_cat_columns_)
        cats = []
        for col, cat in self.cat_map_.items():
            slice_ = self.cat_blocks_[col]
            codes = X[:, slice_].argmax(1)
            series = pd.Series(pd.Categorical.from_codes(
                codes, cat.categories, ordered=cat.ordered
            ), name=col)
            cats.append(series)
        df = pd.concat([non_cat] + cats, axis=1)[self.columns_]
        return df
