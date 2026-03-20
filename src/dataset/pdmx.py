from pathlib import Path

import pandas as pd


class PDMX:
    home: Path

    def __init__(self, home):
        self.home = home
        self.df = pd.read_csv(home / "PDMX.csv")

    def filter(self):
        multitrack = self.df[self.df["n_tracks"] > 2]
        print(f"{len(multitrack)} scroes with more than 2 tracks...")
        print(multitrack[["title", "n_tracks", "path"]].head(20))
