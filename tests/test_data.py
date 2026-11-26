"""Unit tests for the data module's date logic (no network access)."""
import datetime as dt

import pytest

from src import data

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UTC = dt.timezone.utc


@pytest.mark.parametrize(
    "now, expected",
    [
        # Before 16:00 IST today's bar is not final yet, so it is excluded (end is exclusive).
        (dt.datetime(2026, 11, 25, 11, 0, tzinfo=IST), "2026-11-25"),
        (dt.datetime(2026, 11, 25, 15, 59, tzinfo=IST), "2026-11-25"),
        # From 16:00 IST NSE has closed (15:30), so today's bar is included.
        (dt.datetime(2026, 11, 25, 16, 0, tzinfo=IST), "2026-11-26"),
        (dt.datetime(2026, 11, 25, 18, 0, tzinfo=IST), "2026-11-26"),
        # The real scheduled run #126: 21:53 UTC on 25 Nov == 03:23 IST on 26 Nov.
        # Today (26 Nov IST) has no bar yet, so the newest included day is 25 Nov.
        (dt.datetime(2026, 11, 25, 21, 53, tzinfo=UTC), "2026-11-26"),
        # A run at 11:00 UTC (16:30 IST) already sees the finished session.
        (dt.datetime(2026, 11, 26, 11, 0, tzinfo=UTC), "2026-11-27"),
    ],
)
def test_default_end_date_includes_only_finished_sessions(now, expected):
    assert data._default_end_date(now) == expected


def test_default_end_date_uses_the_real_clock_when_none_is_given():
    assert dt.date.fromisoformat(data._default_end_date()) >= dt.datetime.now(IST).date()
