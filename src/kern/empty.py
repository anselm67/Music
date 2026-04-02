"""Base empy handle class for the Humdrum Kern parser.
"""
from typing import List, Optional, Tuple

from kern.parser import Parser
from kern.typing import Token


class EmptySpine:
    pass


class EmptyHandler(Parser[EmptySpine].Handler):

    def open_spine(
        self,
        spine_type: Optional[str] = None,
        parent: Optional[EmptySpine] = None
    ) -> EmptySpine:
        return EmptySpine()

    def close_spine(self, spine: EmptySpine):
        pass

    def branch_spine(self, source: EmptySpine) -> EmptySpine:
        return EmptySpine()

    def merge_spines(self, source: EmptySpine, into: EmptySpine):
        pass

    def rename_spine(self, spine: EmptySpine, name: str):
        pass

    def append(self, tokens: List[Tuple[EmptySpine, Token]]):
        pass

    def done(self):
        pass
