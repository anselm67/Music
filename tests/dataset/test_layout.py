import json

import pytest

from dataset import Box, Page, Score, Staff, System
from utils import from_json


class TestScore:

    def test_save(self):
        system = System(bar_number=1, staves=[
            Staff(box=Box((0, 0), (10, 10)), bars=[1, 2])
        ])
        page = Page(page_number=1, image_width=10, image_height=20,
                    systems=[system],
                    image_rotation=0.0,
                    validated=True)
        score = Score(id='id', pages=[page])
        text = score.asdict()
        saved = from_json(Score, text)
        assert saved == score

    def test_computed_properties(self):
        system = System(bar_number=1, staves=[
            Staff(box=Box((0, 0), (10, 10)), bars=[1, 2]),
            Staff(box=Box((10, 0), (20, 10)), bars=[1, 2])
        ])
        assert system.box == Box((0, 0), (20, 10))
        page = Page(page_number=1, image_width=10, image_height=20,
                    systems=[system],
                    image_rotation=0.0,
                    validated=True)
        assert page.bar_count == sum([x.bar_count for x in page.systems])
