from sheetmusic import Box, Page, Score, Staff, Status, System


class TestScore:
    def test_save(self) -> None:
        system = System(
            bar_numbers=[1],
            bars=[1, 2],
            svg_bar_numbers=[],
            staves=[Staff(top=0, bottom=10)],
        )
        page = Page(
            page_number=1,
            image_width=10,
            image_height=20,
            systems=[system],
            image_rotation=0.0,
            status=Status.VALIDATED,
        )
        score = Score(id="id", pages=[page])
        text = score.asdict()
        saved = Score.from_json(text)
        assert saved == score

    def test_computed_properties(self) -> None:
        system = System(
            bar_numbers=[1],
            bars=[0, 20],
            svg_bar_numbers=[],
            staves=[
                Staff(top=0, bottom=10),
                Staff(top=0, bottom=10),
            ],
        )
        # x from bars[0]..bars[-1]; y is the staff hull.
        assert system.box == Box(0, 0, 20, 10)
        page = Page(
            page_number=1,
            image_width=10,
            image_height=20,
            systems=[system],
            image_rotation=0.0,
            status=Status.VALIDATED,
        )
        assert page.bar_count == sum([x.bar_count for x in page.systems])


class TestScale:
    def test_scale(self) -> None:
        system = System(
            bar_numbers=[1],
            bars=[0, 10],
            svg_bar_numbers=[],
            staves=[Staff(top=0, bottom=10)],
        )
        page = Page(
            page_number=1,
            image_width=10,
            image_height=20,
            systems=[system],
            image_rotation=0.0,
            status=Status.VALIDATED,
        )
        score = Score(id="id", pages=[page])
        scaled = score.resize(20, 40)
        assert scaled.page_count == score.page_count
        for p, s in zip(score.pages, scaled.pages):
            assert p.bar_count == s.bar_count
            assert p.system_count == s.system_count
            for ps, ss in zip(p.systems, s.systems):
                assert ps.bar_count == ss.bar_count
                assert ss.box == Box(0, 0, 20, 20)
                assert ss.bars == [x * 2 for x in ps.bars]
