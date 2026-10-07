import pytest
from precov.survey import horizons

SAMPLE = """API VERSION: 1.2
*******************************************************************************
 Date__(UT)__HR:MN, , , R.A._(ICRF), DEC_(ICRF),
**************************************************************
$$SOE
 2021-Feb-01 08:00, , , 123.456789, -12.345678,
 2021-Feb-02 08:00, , , 124.000000, -12.000000,
$$EOE
**************************************************************
"""


def test_parse_horizons_csv():
    assert horizons.parse_horizons_csv(SAMPLE) == [(123.456789, -12.345678), (124.0, -12.0)]


def test_parse_horizons_error_shows_tail():
    with pytest.raises(RuntimeError, match="no ephemeris"):
        horizons.parse_horizons_csv("blah\nNo ephemeris for target: date out of range")


def test_bad_jd_rejected_before_any_request():
    # the typo from the demo defaults: 22459246.5 / 12459260.5
    with pytest.raises(ValueError, match="JD range"):
        horizons.daily_positions("2021 DG1", 12459260.5, 12459270.5)
