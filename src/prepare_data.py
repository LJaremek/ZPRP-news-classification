from datetime import datetime
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import nltk

nltk.download('vader_lexicon')
nltk.download("punkt")


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

    stop_words = set(stopwords.words("english"))
    sid = SentimentIntensityAnalyzer()

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

    def _sampling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        count_real = df[df["label"] == "real"].shape[0]
        count_fake = df[df["label"] == "fake"].shape[0]
        samples_count = count_real - count_fake

        fake_samples_sampled = df[df["label"] == "fake"].sample(
            n=samples_count,
            replace=True
            )

        df = pd.concat([df, fake_samples_sampled])

        return df

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

    def check_sentiment(self, text: str) -> str:
        scores = self.sid.polarity_scores(text)
        sentiment = max(scores, key=scores.get)
        return sentiment

    def remove_stop_words(self, text: str) -> str:
        words = word_tokenize(text, "english")

        filtered_words = [
            word.lower()
            for word in words
            if word.lower() not in self.stop_words
            ]

        return ' '.join(filtered_words)

    def split_len(self, text: str) -> int:
        return len(text.split(" "))

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
        del df["all_text"]

        df["clear_text"] = df["clear_text"].apply(self.remove_stop_words)

        df["sentiment"] = df["clear_text"].apply(self.check_sentiment)

        df["words_counter"] = df["clear_text"].apply(self.split_len)

        df = self._sampling_data(df)

        df["label"] = df["label"].replace({"fake": 0, "real": 1})

        if result_csv is not None:
            df.to_csv(result_csv)

        return df


if __name__ == "__main__":
    # Example using
    pre_processing = PreProcessing()
    df = pre_processing.process_data(
        "../data/raw/True.csv",
        "../data/raw/Fake.csv",
        "../data/processed/Data2.csv"
        )

    print(df.info())
    print(df.shape)
