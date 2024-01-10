from datetime import datetime
import unittest

import pandas as pd

from .prepare_data import PreProcessing


class BasicTestCase(unittest.TestCase):
    def test_to_datetime(self) -> None:
        pre_processing = PreProcessing()
        self.assertEqual(
            pre_processing._to_datetime("Jun 15, 2024"), datetime(2024, 6, 15)
        )
        self.assertEqual(
            pre_processing._to_datetime("Jun 15, 2024 "), datetime(2024, 6, 15)
        )
        self.assertEqual(
            pre_processing._to_datetime("April 1, 2001"), datetime(2001, 4, 1)
        )
        self.assertEqual(
            pre_processing._to_datetime("April 9, 2101 "), datetime(2101, 4, 9)
        )

    def test_lower_text(self) -> None:
        pre_processing = PreProcessing()
        self.assertEqual(
            pre_processing.lower_text("AlAmAkOtA"), "alamakota"
        )
        self.assertEqual(
            pre_processing.lower_text("Pan Tadeusz"), "pan tadeusz"
        )

    def test_remove_url(self) -> None:
        pre_processing = PreProcessing()
        text = "Visit website at https://www.example.com for more."

        self.assertEqual(
            pre_processing.remove_url(text), "Visit website at  for more."
        )

    def test_remove_bad_symbols(self) -> None:
        pre_processing = PreProcessing()
        text = "It's my #TEXT!! :)) => :D"

        self.assertEqual(
            pre_processing.remove_bad_symbols(text), "Its my TEXT   D"
        )

    def test_clear_text(self) -> None:
        pre_processing = PreProcessing()
        text = "It's my #TEXT!! (REAL     LONG) :)) => :D"

        self.assertEqual(
            pre_processing.clear_text(text), "its my text real long d"
        )

    def test_remove_stop_words(self) -> None:
        pre_processing = PreProcessing()
        text = "The office is a joke or they will not lied as"

        self.assertEqual(
            pre_processing.remove_stop_words(text), "office joke lied"
        )

    def test_sampling_data(self) -> None:
        pre_processing = PreProcessing()

        data = [
            ("A", "real"),
            ("B", "real"),
            ("C", "real"),
            ("D", "fake"),
            ("E", "real"),
            ("F", "fake")
        ]

        df1 = pd.DataFrame(data, columns=("letter", "label"))
        df1 = pre_processing._sampling_data(df1)
        data += [
            ("D", "fake"),
            ("F", "fake")
        ]

        df2 = pd.DataFrame(data, columns=("letter", "label"))

        self.assertEqual(
            df1.shape, df2.shape
        )
        self.assertEqual(
            df1[df1["label"] == "fake"].shape,
            df2[df2["label"] == "fake"].shape
        )


if __name__ == "__main__":
    unittest.main()
