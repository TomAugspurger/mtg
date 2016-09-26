import pytest

import pandas as pd
import pandas.util.testing as tm
from transformer import DummyEncoder

@pytest.fixture
def data():
    df = pd.DataFrame(
        {"A": pd.Categorical(['a', 'b', 'c', 'a'], ordered=True),
         "B": pd.Categorical(['a', 'b', 'c', 'a'], ordered=False),
         "C": pd.Categorical(['a', 'b', 'c', 'a'],
                             categories=['a', 'b', 'c', 'd']),
         "D": [1, 2, 3, 4],
         }
    )
    return df

def test_smoke(data):
    ct = DummyEncoder()
    ct = ct.fit(data)
    trn = ct.transform(data)
    result = ct.inverse_transform(trn)
    # pandas bug on dtype: https://github.com/pydata/pandas/issues/8725
    tm.assert_frame_equal(result, data, check_dtype=False)
