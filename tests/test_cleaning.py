"""Cleaning policies and reporting."""

import numpy as np
import pandas as pd
import pytest

import hydrotrends as ht
from hydrotrends.core.constants import COL_DATE, COL_FLOW_CUSECS
from hydrotrends.core.exceptions import ValidationError


def _messy():
    df = pd.DataFrame(
        {
            COL_DATE: pd.to_datetime(
                ["2000-04-01", "2000-04-02", "2000-04-02", "2000-04-03", "2000-04-04"]
            ),
            COL_FLOW_CUSECS: [100.0, np.nan, 200.0, -50.0, "bad"],
        }
    )
    return df


def test_default_policy_drops_missing_and_dupes_keeps_negatives():
    cleaned, rep = ht.clean(_messy())
    assert rep.n_missing == 2 and rep.n_duplicate_dates == 1 and rep.n_negative == 1
    assert rep.n_removed == 3
    assert (cleaned[COL_FLOW_CUSECS] < 0).any()  # negative kept


def test_drop_negatives():
    _, rep = ht.clean(_messy(), ht.CleaningPolicy(negatives="drop"))
    assert rep.n_removed == 4


@pytest.mark.parametrize(
    "policy",
    [{"missing": "error"}, {"duplicate_dates": "error"}, {"negatives": "error"}],
)
def test_error_policy_raises(policy):
    with pytest.raises(ValidationError):
        ht.clean(_messy(), ht.CleaningPolicy(**policy))


def test_bad_policy_rejected():
    with pytest.raises(ValidationError):
        ht.CleaningPolicy(missing="nuke")


def test_linear_fill_interior_only():
    df = pd.DataFrame(
        {
            COL_DATE: pd.to_datetime([f"2000-04-0{i}" for i in range(1, 8)]),
            COL_FLOW_CUSECS: [100.0, np.nan, np.nan, 400.0, np.nan, 600.0, np.nan],
        }
    )
    cleaned, rep = ht.clean(df, ht.CleaningPolicy(missing_fill="linear"))
    assert rep.n_filled == 3  # interior gaps only
    assert rep.n_missing == 1  # trailing NaN remains -> dropped by default
    got = cleaned.set_index(COL_DATE)[COL_FLOW_CUSECS]
    assert got[pd.Timestamp("2000-04-02")] == 200.0
