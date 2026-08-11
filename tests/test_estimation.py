import pandas as pd
from hai_funsearch.estimation.empirical import calibrate_promise_by_bins


def test_empirical_calibration_range():
    df = pd.DataFrame({"promise_raw": [0.1,0.2,0.8,0.9], "improved": [0,0,1,1]})
    out, cal = calibrate_promise_by_bins(df, bins=2)
    assert out.promise.between(0,1).all()
    assert len(cal) == 2
