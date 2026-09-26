"""Tests for tracking the hero's hole cards across a hand.

Reading the fanned hole cards is expensive (~114ms, sometimes far more) and
they only change once a hand, so the read happens at preflop and the answer is
held. Two separate conditions govern that, and keeping them separate is the
whole point:

  * WHEN MAY WE READ -- only with an empty board, i.e. preflop.
  * WHEN DOES THE HELD ANSWER STOP BEING TRUE -- when the hero card region
    itself changes. Measured on real frames, ordinary animation moves that
    region by 4-468 while a new deal moves it by 9471-9772, a 20x gap.

Using the board alone for both was tried and is wrong: during a run-it-once
showdown the board renders elsewhere, reads as empty, and a mid-hand re-read
picked up the wrong cards and held them for the rest of the hand.
"""
from poker_vision.hero_cards import HeroCardTracker

SAME = [0] * 16
OTHER = [200] * 16


def _reader(*results):
    queue = list(results)
    calls = []

    def read():
        calls.append(True)
        return queue.pop(0) if queue else None

    read.calls = calls
    return read


def test_reads_the_cards_at_preflop():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()

    assert tracker.update([], read, SAME) == ["Ad", "Kd"]
    assert len(read.calls) == 1


def test_holds_them_once_the_flop_lands():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)

    assert tracker.update(["3c", "6c", "6d"], read, SAME) == ["Ad", "Kd"]
    assert len(read.calls) == 1


def test_does_not_reread_across_a_long_preflop():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    for _ in range(5):
        tracker.update([], read, SAME)

    assert len(read.calls) == 1


def test_a_board_that_briefly_misreads_as_empty_does_not_trigger_a_reread():
    # The run-it-once showdown renders the board elsewhere and it reads as
    # empty. The cards on screen have not changed, so neither should ours.
    read = _reader(["Ad", "Kd"], ["4s", "Qs"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)
    tracker.update(["Ts", "Qd", "2s"], read, SAME)

    assert tracker.update([], read, SAME) == ["Ad", "Kd"]
    assert len(read.calls) == 1


def test_new_cards_on_screen_trigger_a_fresh_read():
    read = _reader(["Ad", "Kd"], ["7s", "2h"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)

    assert tracker.update([], read, OTHER) == ["7s", "2h"]
    assert len(read.calls) == 2


def test_new_cards_with_no_preflop_frame_show_nothing_rather_than_the_old_ones():
    # The capture gate can skip preflop entirely. Holding the previous hand's
    # cards would show the wrong hand with no sign of it; blank is the failure
    # someone actually notices.
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)

    assert tracker.update(["9h", "2s", "Kc"], read, OTHER) == []
    assert len(read.calls) == 1        # never read outside preflop


def test_starting_mid_hand_shows_nothing_until_the_next_deal():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()

    assert tracker.update(["3c", "6c", "6d"], read, SAME) == []
    assert read.calls == []


def test_keeps_trying_through_preflop_until_a_read_succeeds():
    read = _reader(None, [], ["Ad", "Kd"])
    tracker = HeroCardTracker()

    assert tracker.update([], read, SAME) == []
    assert tracker.update([], read, SAME) == []
    assert tracker.update([], read, SAME) == ["Ad", "Kd"]


def test_sitting_out_shows_no_cards():
    read = _reader(None, None)
    tracker = HeroCardTracker()

    assert tracker.update([], read, SAME) == []
    assert tracker.update(["3c", "6c", "6d"], read, SAME) == []


def test_works_without_a_signature_by_falling_back_to_the_board():
    # Older payloads carry no hero strip, so there is nothing to compare.
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()

    assert tracker.update([], read, None) == ["Ad", "Kd"]


def test_hole_cards_that_clash_with_the_board_are_dropped():
    # A card cannot be in the board and your hand at once. When both readers
    # claim it, one is wrong and there is no way to tell which -- showing the
    # pair anyway presents an impossible hand as fact. Found by backtesting:
    # the board's ace of spades read as a diamond and collided with the hero's
    # real ace of diamonds.
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)

    assert tracker.update(["5s", "8h", "Qd", "4s", "Ad"], read, SAME) == []


def test_hole_cards_are_kept_when_the_board_does_not_clash():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)

    assert tracker.update(["5s", "8h", "Qd"], read, SAME) == ["Ad", "Kd"]


def test_a_clash_appearing_later_still_drops_them():
    read = _reader(["Ad", "Kd"])
    tracker = HeroCardTracker()
    tracker.update([], read, SAME)
    tracker.update(["5s", "8h", "Qd"], read, SAME)

    assert tracker.update(["5s", "8h", "Qd", "Ad"], read, SAME) == []


def test_a_half_read_hand_is_not_latched():
    # The tracker latches the first non-empty read of a hand and then stops
    # reading, so latching a single card shows half a hand for the rest of
    # it, with no second chance. Measured across the live corpora, 22
    # pre-flop frames read exactly one card against 191 that read both, so
    # waiting costs a frame and buys the other card.
    tracker = HeroCardTracker()
    reads = [["Ad"], ["Ad", "Kc"]]

    first = tracker.update([], read=lambda: reads[0], signature=[0])
    assert first == [], "half a hand was shown as if it were the hand"

    second = tracker.update([], read=lambda: reads[1], signature=[0])
    assert second == ["Ad", "Kc"]


def test_a_complete_pair_is_still_latched_at_once():
    tracker = HeroCardTracker()

    assert tracker.update([], read=lambda: ["Ad", "Kc"], signature=[0]) == ["Ad", "Kc"]
    # and held once the board arrives, without reading again
    assert tracker.update(["2h", "7d", "9s"],
                          read=lambda: (_ for _ in ()).throw(AssertionError("re-read")),
                          signature=[0]) == ["Ad", "Kc"]


def test_two_of_the_same_card_is_not_a_hand():
    # Impossible, and seen live: a dimmed jack read as a four beside a real
    # four gave ["4s", "4s"]. Whichever card is wrong, the pair is not real.
    tracker = HeroCardTracker()

    assert tracker.update([], read=lambda: ["4s", "4s"], signature=[0]) == []
