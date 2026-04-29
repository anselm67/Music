"""Main class for accessing the PDMX dataset.

This class manages tghe patgh of files within a PDMX dataset, in addition:
- It generates the files needed for training models,
- It can computes various statistics about the underlying dataset.
"""
import json
import os
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from utils import compile_filter

from .layout import Score


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.exists() and dst_file.stat().st_mtime >= src_file.stat().st_mtime


type DirClass = Literal[
    # Original dir classes form PDMX tar file.
    'metadata', 'mxl', 'pdf', 'data',
    # Derived one, will leave under PDNX/build.
    'krn', 'tokens', 'layout',  'svg', 'png'
]


class PDMX:
    EXTENSIONS: dict[DirClass, str] = {
        'data': '.json',
        'krn': '.krn',
        'tokens': '.tokens',
        'layout': '.json',
        'metadata': '.json',
        'mxl': '.mxl',
        'pdf': '.pdf',
        'svg': '.svg',
        'png': '.png'
    }
    CSV_SCHEMA = {
        "path": "Path to the data (MusicRender JSON) file.",
        "metadata": "Path to the associated metadata (JSON) file. The basename of each file matches the basename of the corresponding file in the path column.",
        "mxl": "Path to the associated compressed MusicXML (MXL) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to compressed MusicXML.",
        "pdf": "Path to the associated sheet music (PDF) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to PDF.",
        "mid": "Path to the associated MIDI (MID) file. The basename of each file matches the basename of the corresponding file in the path column. Values may be N/A, since some of the original MuseScore files are corrupted and thus cannot be converted to MID.",
        "version": "Version of the original MuseScore file.",
        "is_user_pro": "Whether the user who posted the original MuseScore file is a 'pro' user (pays for a MuseScore subscription).",
        "is_user_publisher": "Whether the user who posted the original MuseScore file is a music publisher.",
        "is_user_staff": "Whether the user who posted the original MuseScore file is part of MuseScore staff.",
        "has_paywall": "Whether the original MuseScore file had a paywall.",
        "is_rated": "Whether the original MuseScore file had any ratings.",
        "is_official": "Whether the original MuseScore file was an 'official' score, a title decided by MuseScore.",
        "is_original": "Whether the original MuseScore file was an original work.",
        "is_draft": "Whether the original MuseScore file was marked as a draft by the user who posted it.",
        "has_custom_audio": "Whether the original MuseScore file has an associated custom audio file (must retrieve the actual audio from the metadata).",
        "has_custom_video": "Whether the original MuseScore file has an associated custom video file (must retrieve the actual video from the metadata).",
        "n_comments": "Number of comments on the original MuseScore file.",
        "n_favorites": "Number of users who favorited the original MuseScore file.",
        "n_views": "Number of views on the original MuseScore file.",
        "n_ratings": "Number of ratings on the original MuseScore file.",
        "rating": "Average rating (out of five stars) of the original MuseScore file. A rating of zero indicates that a song is unrated.",
        "license": "Creative Commons license of the original MuseScore file.",
        "license_url": "Link to the Creative Commons license of the original MuseScore file. Directly related to the license column.",
        "license_conflict": "Whether the song's public-facing copyright metadata license disagrees with the internal copyright license data of the original MuseScore file.",
        "genres": "Genre(s) associated with the original MuseScore file, separated with a '-' if there are multiple.",
        "groups": "MuseScore group(s) associated with the original MuseScore file, separated with a '-' if there are multiple.",
        "tags": "MuseScore tag(s) associated with the original MuseScore file, separated with a '-' if there are multiple.",
        "song_name": "If available, the name of the song.",
        "title": "If available, the title of the song, oftentimes the same as song_name.",
        "subtitle": "If available, the subtitle of the song.",
        "artist_name": "If available, the name of the artist who created the song.",
        "composer_name": "If available, the name of the composer who created the song, oftentimes the same as artist_name.",
        "publisher": "If available, the publisher of the song.",
        "complexity": "The MuseScore complexity score of the original MuseScore file. Ranges from 0-3.",
        "n_tracks": "The number of tracks (parts) in the original MuseScore file.",
        "tracks": "Track{s} from the original MuseScore file, separated with a '-' if there are multiple.",
        "song_length": "Length of the song, in metrical MusPy time steps.",
        "song_length.seconds": "Length of the song, in seconds.",
        "song_length.bars": "Length of the song, in bars.",
        "song_length.beats": "Length of the song, in beats.",
        "n_notes": "Number of notes in the song.",
        "notes_per_bar": "Average number of notes per bar in the song.",
        "n_annotations": "Number of annotations in the song.",
        "has_annotations": "Whether the song has any annotations.",
        "n_lyrics": "Number of lyrics in the song.",
        "has_lyrics": "Whether the song has any lyrics.",
        "n_tokens": "Number of tokens (n_notes + n_annotations + n_lyrics) in the song.",
        "pitch_class_entropy": "Pitch Class Entropy of the song, as calculated by the MusPy Package.",
        "scale_consistency": "Scale Consistency of the song, as calculated by the MusPy Package. Ranges from 0-1.",
        "groove_consistency": "Groove Consistency of the song, as calculated by the MusPy Package. Ranges from 0-1.",
        "best_path": "Best filepath in the song's title duplicate grouping (see paper for full description).",
        "is_best_path": "Whether the song is the best_path in the title duplicate grouping.",
        "best_arrangement": "Best filepath in the song's title-instrumentation duplicate grouping.",
        "is_best_arrangement": "Whether the song is the best_arrangement in the title-instrumentation duplicate grouping.",
        "best_unique_arrangement": "Best filepath in the song's title-instrumentation-arrangement duplicate grouping.",
        "is_best_unique_arrangement": "Whether the song is the best_unique_arrangement in the title-instrumentation-arrangement duplicate grouping. All songs for which this value is true are part of the Deduplicated subset.",
        "subset:all": "Whether the song is part of the All subset (all True).",
        "subset:deduplicated": "Whether the song is part of the Deduplicated subset (the same as is_best_unique_arrangement).",
        "subset:rated": "Whether the song is part of the Rated subset (the song has a non-zero rating).",
        "subset:rated_deduplicated": "Whether the song is both part of the Rated and Deduplicated subsets.",
        "subset:no_license_conflict": "Whether the song's public-facing copyright metadata license agrees with the internal copyright license data of the original MuseScore file (the negation of the license_conflict column).",
        "subset:all_valid": "Whether the song's associated compressed MusicXML (MXL), sheet music (PDF), and MIDI (MID) files are all valid (non-N/A)."
    }
    home: Path

    def __init__(self, home, name: str = "PDMX.csv", offset: int = -1, count: int = -1):
        self.home = home
        self.df = pd.read_csv(home / name)
        self.slice(offset, count)

    def __len__(self) -> int:
        return len(self.df)

    def slice(self, offset: int, count: int) -> Self:
        offset = max(0, min(offset, len(self.df)))
        count = min(count, len(self.df) - offset)
        if count < 0:
            self.df = self.df.iloc[offset:]
        else:
            self.df = self.df.iloc[offset: offset + count]
        return self

    def relative(self, path) -> Path:
        return path.relative_to(self.home)

    def get_path(self, some: Path, dir_class: DirClass, mkdirs: bool = False) -> Path:
        relative = self.relative(some) if some.is_absolute() else some
        # Strips the optional 'build' component of the path.
        if len(relative.parts) >= 1 and relative.parts[0] == "build":
            relative = Path(*relative.parts[1:])
        # Strips the dirclass component.
        if len(relative.parts) <= 1:
            raise ValueError(f"Unexpected path structure: {some}")
        relative = Path(*relative.parts[1:])
        # Compose the path, under 'build' if dirclass isn't original.
        if dir_class in ['metadata', 'mxl', 'pdf', 'data']:
            path = (self.home / dir_class /
                    relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        else:
            path = (self.home / "build" / dir_class /
                    relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        if mkdirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_page_path(self, some: Path, dir_class: DirClass, page_number: int) -> Path:
        path = self.get_path(some, dir_class)
        stem = f"{path.stem}_{page_number:03d}"
        return path.with_stem(stem)

    def get_err_path(self, path: Path) -> Path:
        return path.with_suffix('.err')

    def touch_err_path(self, path: Path):
        err_path = self.get_err_path(path)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.touch()

    def query(self, query_string, metadata: str | None, score: str | None) -> pd.DataFrame:
        metadata_filter, score_filter = (
            compile_filter(metadata) if metadata else None,
            compile_filter(score) if score else None,
        )

        df = self.df.query(query_string)
        if metadata_filter is not None or score_filter is not None:
            def filter_row(row) -> bool:
                if not isinstance(row['mxl'], str) or not isinstance(row['metadata'], str):
                    return False
                try:
                    if metadata_filter is not None:
                        metadata_file = (self.home / row['metadata'])
                        obj = json.loads(metadata_file.read_text())
                        if not metadata_filter(obj):
                            return False
                    if score_filter is not None:
                        layout_file = self.get_path(
                            (self.home / row['mxl']), 'layout')
                        obj = Score.from_json(
                            json.loads(layout_file.read_text()))
                        if not score_filter(obj):
                            return False
                    return True
                except (FileNotFoundError, json.JSONDecodeError):
                    return False
            return df[df.apply(filter_row, axis=1)]
        else:
            return df

    def make(self,
             mxl_file: Path | None = None,
             num_workers: int | None = None,
             force: bool = False,
             dry_run: bool = False
             ):
        from .pdmx_maker import PDMXMaker
        if num_workers is None:
            num_workers = os.cpu_count() or 4
        maker = PDMXMaker(self, force=force, dry_run=dry_run)
        maker.run(mxl_file, num_workers)

    def stats(self, num_worker: int | None = None):
        from .pdmx_stater import PDMXStater
        if num_worker is None:
            num_worker = os.cpu_count() or 4
        stater = PDMXStater(self)
        return stater.run(num_worker)

    def info(self, mxl_file: Path) -> list[tuple[str, str]] | None:
        value = mxl_file.name
        for _, row in self.df.iterrows():
            mxl_str = row['mxl']
            if not isinstance(mxl_str, str):
                continue
            elif Path(mxl_str).name == value:
                infos: list[tuple[str, str]] = list()
                for col in self.df.columns:
                    infos.append(
                        (self.CSV_SCHEMA.get(col) or col, row[col])
                    )
                return infos
        return None

    def pick_mxl(self) -> Path:
        while True:
            row = self.df.sample(n=1).iloc[0]
            mxl_str = row['mxl']
            if isinstance(mxl_str, str):
                return self.get_path(Path(mxl_str), 'mxl')

# vscode - End of File
