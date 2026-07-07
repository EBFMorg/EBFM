# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import unittest

from datetime import datetime
import isodate
from argparse import Namespace
from ebfm.core.config import TimeConfig


class TestTimeConfig(unittest.TestCase):
    """Exactly one primary grid option must be provided."""

    def test_init(self):
        args = Namespace(
            start_time="2024-01-01T00:00:00+00:00",
            # start_time="2024-01-01T00:00:00Z",  # only for Python 3.11 and later, not Python 3.10
            end_time="2024-01-02T00:00:00+00:00",
            # start_time="2024-01-02T00:00:00Z",  # only for Python 3.11 and later, not Python 3.10
            time_step="PT1H",
            calendar="proleptic_gregorian",
        )

        time_config = TimeConfig(args)
        self.assertEqual(time_config.start_time, datetime.fromisoformat(args.start_time))
        self.assertEqual(time_config.end_time, datetime.fromisoformat(args.end_time))
        self.assertEqual(time_config.time_step, isodate.parse_duration(args.time_step))
        self.assertEqual(time_config.calendar.value, args.calendar)

    def test_init_no_tz(self):
        """
        Test that default timezone is automatically added if no timezone is given.
        """

        from ebfm.core.config import DEFAULT_TZ

        args = Namespace(
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            time_step="PT1H",
            calendar="proleptic_gregorian",
        )

        time_config = TimeConfig(args)
        self.assertEqual(time_config.start_time, datetime.fromisoformat(args.start_time).replace(tzinfo=DEFAULT_TZ))
        self.assertEqual(time_config.end_time, datetime.fromisoformat(args.end_time).replace(tzinfo=DEFAULT_TZ))
        self.assertEqual(time_config.time_step, isodate.parse_duration(args.time_step))
        self.assertEqual(time_config.calendar.value, args.calendar)


if __name__ == "__main__":
    unittest.main()
