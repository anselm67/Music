import unittest
from typing import List, Optional, Tuple
from unittest.mock import Mock, call

from kern.parser import Parser
from kern.typing import Bar, Duration, Note, Pitch, Rest, Token


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


class TestHumdrumParser(unittest.TestCase):

    def ok(self, text: str):
        parser = Parser.from_text(text, EmptyHandler())
        parser.parse()

    def fail(self, text: str):
        parser = Parser.from_text(text, EmptyHandler())
        with self.assertRaises(ValueError):
            parser.parse()

    def parse_one_token(self, text: str, expected_token: Token):
        mock_handler = Mock()
        handler_instance = mock_handler.return_value
        parser = Parser.from_text(
            "**kern\n" + text + "\n",
            handler_instance
        )
        parser.parse()
        self.assertEqual(handler_instance.open_spine.call_count, 1)
        handler_instance.append.assert_has_calls([call([(
            handler_instance.open_spine.return_value,
            expected_token
        )])])

    def test_kerns(self):
        self.fail("")
        self.ok(
            "**kern\t**kern\n"
            "*clefF4\t*clefG2"
        )

    def test_spine_indicators(self):
        self.ok(
            "**kern\t**kern\n"
            "*-\t*"
        )

    def test_suggested_duration_note(self):
        # joplin/elite.krn has a lot of this(!)
        self.ok(
            "**kern\n"
            "4C C\n"
        )

    def test_handler_called(self):
        mock_handler = Mock()
        handler_instance = mock_handler.return_value
        parser = Parser.from_text(
            "**kern\t**kern\n",
            handler_instance
        )
        parser.parse()
        self.assertEqual(handler_instance.open_spine.call_count, 2)

    def test_note_parsing(self):
        mock_handler = Mock()
        handler_instance = mock_handler.return_value
        parser = Parser.from_text(
            "**kern\n"
            "8A\n",
            handler_instance
        )
        parser.parse()
        self.assertEqual(handler_instance.open_spine.call_count, 1)
        handler_instance.append.assert_has_calls([call([
            (handler_instance.open_spine.return_value, Note(
                pitch=Pitch.A,
                duration=Duration(8)
            ))])])

    def test_some_tokens(self):
        self.parse_one_token("8A\n", Note(
            pitch=Pitch.A,
            duration=Duration(8)
        ))
        self.parse_one_token("8A-\n", Note(
            pitch=Pitch.A,
            duration=Duration(8),
            flats=1,
        ))
        self.parse_one_token("8A##LL\n", Note(
            pitch=Pitch.A,
            duration=Duration(8),
            sharps=2,
            starts_beam=2
        ))

    def test_note_duration(self):
        self.parse_one_token("8.A\n", Note(
            pitch=Pitch.A,
            duration=Duration(8, 1),
        ))
        self.parse_one_token("16..A\n", Note(
            pitch=Pitch.A,
            duration=Duration(16, 2),
        ))

    def test_rest_duration(self):
        self.parse_one_token("8r\n", Rest(
            duration=Duration(8, 0),
        ))
        self.parse_one_token("16..r\n", Rest(
            duration=Duration(16, 2),
        ))

    def test_rest_duration_extra(self):
        self.parse_one_token("8ryy\n", Rest(
            duration=Duration(8, 0),
        ))

    def test_open_before_note_token(self):
        self.parse_one_token("(16..A\n", Note(
            pitch=Pitch.A,
            duration=Duration(16, 2),
            starts_slur=True,
        ))

    def test_ritardendo_note_token(self):
        # https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/chopin/mazurka&file=mazurka06-1.krn&format=info
        self.parse_one_token("(20%3A#\n", Note(
            pitch=Pitch.A,
            sharps=1,
            duration=Duration(3, 0),
            starts_slur=True,
        ))

    def test_barred_gracenote_token(self):
        # https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/chopin/mazurka&file=mazurka06-1.krn&format=info
        self.parse_one_token("(<8qgg#/\n", Note(
            pitch=Pitch.gg,
            sharps=1,
            duration=Duration(8, 0),
            starts_slur=True,
            is_gracenote=True
        ))

    def test_wrapped_note_token(self):
        # https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/chopin/mazurka&file=mazurka06-1.krn&format=info
        self.parse_one_token("&(4B#&)\n", Note(
            pitch=Pitch.B,
            sharps=1,
            starts_slur=True,
            ends_slur=True,
            duration=Duration(4, 0),
        ))

    def test_thrilled_note(self):
        self.parse_one_token("4anT^\n", Note(
            pitch=Pitch.a,
            sharps=0, flats=0,
            duration=Duration(4, 0),
            is_upper_thrill=True
        ))
        self.parse_one_token("4ant^\n", Note(
            pitch=Pitch.a,
            sharps=0, flats=0,
            duration=Duration(4, 0),
            is_lower_thrill=True
        ))

    def test_random_stuff_i_ve_run_into(self):
        self.parse_one_token("[</2b-\n", Note(
            pitch=Pitch.b,
            flats=1,
            duration=Duration(2, 0),
            starts_tie=True,
        ))
        self.parse_one_token("(16qqbbP\n", Note(
            pitch=Pitch.bb,
            duration=Duration(16, 0),
            starts_slur=True,
            is_gracenote=True
        ))
        self.parse_one_token("(8qqPee\n", Note(
            pitch=Pitch.ee,
            duration=Duration(8, 0),
            starts_slur=True,
            is_gracenote=True
        ))

    def test_some_asap_dataset_tokens(self):
        self.parse_one_token(".ZZZ16g#LL\n", Note(
            pitch=Pitch.g,
            duration=Duration(16, 0),
            sharps=1,
            starts_beam=2,
        ))
        self.parse_one_token("8%-3ryy\n", Rest(
            duration=Duration(3, 0),
        ))

    def test_bar_number(self):
        self.parse_one_token("= 7 \n", Bar(
            "= 7", 7,
            False, False, False, False))
        self.parse_one_token("==\n", Bar(
            "==", barno=-1,
            is_final=True,
            is_invisible=False,
            is_repeat_start=False, is_repeat_end=False))


if __name__ == '__main__':
    unittest.main()
