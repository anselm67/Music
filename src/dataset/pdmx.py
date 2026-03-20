from pathlib import Path

import pandas as pd


class PDMX:
    home: Path

    def __init__(self, home):
        self.home = home
        self.df = pd.read_csv(home / "PDMX.csv")

    def query(self, query_string) -> pd.DataFrame:
        return self.df.query(query_string)
