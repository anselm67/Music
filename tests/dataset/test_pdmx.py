from pathlib import Path

import pytest

from dataset import PDMX


class TestPDMX:

    def test_get_path(self, tmp_path: Path):
        (tmp_path / 'PDMX.csv').write_text("id\n1")
        pdmx = PDMX(tmp_path)
        assert pdmx.get_path(Path(tmp_path) / 'mxl' / 'abc',
                             'svg') == Path(tmp_path) / 'svg' / 'abc.svg'
        assert pdmx.get_path(Path(tmp_path) / 'svg' / 'abc.svg',
                             'mxl') == Path(tmp_path) / 'mxl' / 'abc.mxl'

        with pytest.raises(ValueError):
            pdmx.get_path(Path(tmp_path), 'mxl')
