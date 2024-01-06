from datetime import datetime
import re

import pandas as pd


class PreProcessing:
    DATE_FORMATS = (
        "%B %d, %Y ",
        "%B %d, %Y",
        "%d-%B-%y",
        "%b %d, %Y ",
        "%b %d, %Y",
        "%d-%b-%y"
    )
    BAD_SYMBOLS = (
        '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
        '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
        '{', '|', '}', '~'
    )

    def lower_text(self, text: str) -> str:
        return text.lower()

    def remove_html(self, text: str):
        return re.sub("()", "", text, flags=re.DOTALL)

    def remove_url(self, text: str) -> str:
        return re.sub(r'https?:\/\/.\S+', "", text)

    def remove_brackets(self, text: str) -> str:
        text = re.sub("\\[]", "", text)
        text = re.sub("\\(\\)", "", text)
        text = re.sub("\\{}", "", text)
        text = re.sub("\\]", "", text)
        text = re.sub("\\[", "", text)
        return text

    def remove_bad_symbols(self, text: str) -> str:
        for bad_symbol in self.BAD_SYMBOLS:
            text = text.replace(bad_symbol, "")
        return text

    def clear_text(self, text: str) -> str:
        text = self.lower_text(text)
        text = self.remove_html(text)
        text = self.remove_url(text)
        text = self.remove_brackets(text)
        text = self.remove_bad_symbols(text)

        text = text.replace("\n", " ")

        old_text = ""
        while old_text != text:
            old_text = text
            text = text.replace("  ", " ")

        return text

    def _to_datetime(self, date_str: str) -> datetime:
        """
        Unification of the date format.
        Based on the DATE_FORMATS class variable:
        "%B %d, %Y " -> datetime
        "%B %d, %Y"  -> datetime
        "%d-%b-%y"   -> datetime
        ...

        Input:
         * date_str: str - invalid date format as a string

        Output:
         * dt: datetime - datetime type
        """
        result = None

        for format in self.DATE_FORMATS:
            try:
                result = pd.to_datetime(date_str, format=format)
            except ValueError:
                pass

            if result is not None:
                break

        return result

    def _process_dataframe(
            self,
            df: pd.DataFrame,
            data_type: str
            ) -> pd.DataFrame:
        """
        Input:
         * df: pd.DataFrame
         * data_type: str - 'real' or 'fake'
        Output:
         * df: pd.DataFrame - prepared pandas DataFrame
        """

        df["label"] = data_type

        df["new_date"] = df["date_index"] = df["date"].apply(self._to_datetime)
        df.set_index("date_index", inplace=True)

        df = df.drop_duplicates(subset=["text"], ignore_index=True)

        df["all_text"] = df["title"] + " " + df["text"]
        del df["title"]
        del df["text"]

        return df

    def process_data(
            self,
            real_csv: str,
            fake_csv: str,
            result_csv: str | None = None
            ) -> pd.DataFrame:
        """
        Process data and prepare them for the model:
         - unification of the date format
         - cleaning the text
         - getting rid of duplicates
         - adding labels

        Input:
         * real_csv: str - path to csv file with real data
         * fake_csv: str - path to csv file with fake data
         * result_csv: str | None - path for the result data frame. If you do
            not want to save the results as a file, miss this.

        Output:
         * df: pd.DataFrame - result data as a pandas DataFrame
        """

        df_real = self._process_dataframe(pd.read_csv(real_csv), "real")
        df_fake = self._process_dataframe(pd.read_csv(fake_csv), "fake")

        df = pd.concat([df_real, df_fake])
        df.reset_index(drop=True, inplace=True)

        df["clear_text"] = df["all_text"].apply(self.clear_text)

        if result_csv is not None:
            df.to_csv(result_csv)

        return df


if __name__ == "__main__":
    pre_processing = PreProcessing()
    pre_processing.process_data(
        "data/True.csv",
        "data/Fake.csv",
        "data/Data2.csv"
        )
