import unittest

import scraper


class BasicTestCase(unittest.TestCase):
    def test_extracts_longest(self):
        page = """
        <p>Lorem ipsum dolor sit amet</p>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
        <p></p>
        """
        article = scraper.extract_article(page)
        self.assertEqual(
            article, "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )

    def test_extracts_nested(self):
        page = """
        <p>Lorem ipsum dolor sit amet</p>
        <div>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
        </div>
        <p></p>
        """
        article = scraper.extract_article(page)
        self.assertEqual(
            article, "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )

    def test_fails_on_missing_article(self):
        page = """
        <p></p>
        <div>
            <p></p>
        </div>
        <p></p>
        """
        article = scraper.extract_article(page)
        self.assertEqual(article, None)


if __name__ == "__main__":
    unittest.main()
