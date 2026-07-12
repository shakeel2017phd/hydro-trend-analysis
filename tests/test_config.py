"""Config / InputSpec validation and serialisation."""
import pytest

import hydrotrends as ht
from hydrotrends.core.exceptions import ConfigError


def test_defaults_build_both_schemes_and_units():
    cfg = ht.Config()
    assert cfg.includes_cropping and cfg.includes_meteorological
    assert cfg.flow_units == (ht.FlowUnit.CUSECS, ht.FlowUnit.CUMECS)


@pytest.mark.parametrize("bad", [
    {"alpha": 1.5}, {"alpha": 0.0}, {"dpi": 0}, {"lowess_frac": 0.0},
    {"flow_units": ["Furlongs"]}, {"season_schemes": []},
    {"season_schemes": ["monsoon"]},
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ConfigError):
        ht.Config(**bad)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        ht.Config.from_dict({"alpah": 0.1})


def test_input_spec_value_scale_and_date_format():
    spec = ht.InputSpec("x.csv", "10daily", value_column="Inflow_1000Cusecs",
                        value_scale=1000, date_format="dekad_compact")
    assert spec.value_scale == 1000.0
    assert spec.date_format is ht.DateFormat.DEKAD_COMPACT


def test_dekad_compact_requires_10daily():
    with pytest.raises(ConfigError):
        ht.InputSpec("x.csv", "daily", date_format="dekad_compact")


@pytest.mark.parametrize("bad", [{"value_scale": 0}, {"resolution": "hourly"},
                                 {"date_format": "julian"}])
def test_bad_input_spec_rejected(bad):
    with pytest.raises(ConfigError):
        ht.InputSpec("x.csv", bad.pop("resolution", "daily"), **bad)


def test_yaml_round_trip(tmp_path):
    cfg = ht.Config(inputs=[ht.InputSpec("d.csv", "daily"),
                            ht.InputSpec("t.csv", "10daily", value_scale=1000)])
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    assert ht.Config.from_yaml(path).to_dict() == cfg.to_dict()
