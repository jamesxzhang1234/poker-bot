"""Tracks the hero's hole cards across a hand.

Reading the fanned hole cards costs ~114ms and occasionally far more, because
the cards are tilted and overlapping so the rank has to be hunted for. They
also only change once a hand. So the read happens at preflop and the answer is
held -- with two separate conditions governing it, and keeping them separate
is the whole design:

  * WHEN MAY WE READ -- only when the board is empty, i.e. preflop, which is
    when a new pair has just been dealt.
  * WHEN DOES THE HELD ANSWER STOP BEING TRUE -- when the hero card region
    itself changes. Measured across real frames, ordinary animation moves that
    region by 4-468 while a new deal moves it by 9471-9772: a 20x gap with
    nothing in between.

Using the board for both was tried and is wrong. During a run-it-once showdown
the board renders somewhere else, the fixed slots read it as empty, and that
false preflop triggered a mid-hand re-read which picked up the wrong cards and
held them for the rest of the hand.

This is the only part of the pipeline that remembers anything between frames,
which CLAUDE.md otherwise rules out. It earns the exception by reasoning only
about whether a value it read is still valid -- never inferring table state --
and by discarding that value the moment it cannot show it still applies.
Holding the previous hand's cards into a new hand would display the wrong hand
with nothing to suggest it, and those are the two cards a person trusts most,
so when in doubt it shows none.
"""

# Sum of absolute differences over a 16x16 grey thumbnail of the hero card
# region. Real deals measured 9471-9772 and ordinary animation 4-468, so this
# sits in an empty band an order of magnitude from either side.
CHANGE_THRESHOLD = 2000


# A hold'em hand is two distinct cards. Anything else is a half-read or an
# impossible one, and latching either shows it for the rest of the hand --
# the tracker stops reading once it has an answer, so there is no second
# chance. Measured across the live corpora, 22 pre-flop frames read exactly
# one card against 191 that read both, so waiting costs a frame and buys the
# missing card. Two of the same card has been seen too: a dimmed jack read
# as a four, beside a real four.
HOLE_CARDS = 2


def _is_a_hand(cards):
    return len(cards) == HOLE_CARDS and len(set(cards)) == HOLE_CARDS


def _changed(a, b):
    if a is None or b is None or len(a) != len(b):
        return True
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b)) > CHANGE_THRESHOLD


# Consecutive pre-flop reads that must agree before a hand is shown, and
# consecutive reads that must contradict a shown hand before it is withdrawn.
#
# A single read was trusted before, and checked against 24 hands labelled by
# eye that put a wrong card on screen in 6 of them. Most were the rank reader
# (fixed separately), but two were one bad frame followed by good ones -- an
# 8 read as a 3 on the first pre-flop frame only -- and the first read was
# the one kept. Reads cost ~9ms now that Tesseract stays in-process, so the
# tracker reads on every pre-flop frame instead of once.
AGREE = 2


class HeroCardTracker:
    """Shows the hero's hole cards once two reads agree; never shows a doubt.

    Pre-flop, every frame is read. Nothing is shown until two consecutive
    reads give the same two cards. Once shown, a contradicting read hides the
    hand at once -- a doubted hand is not shown -- and an agreeing one brings
    it back, since single-frame misreads happen. Two contradictions in a row
    settle it, one of two ways:

      * No board card has appeared since the hand was shown: we are still in
        the same pre-flop, so this is a deal the signature missed. A live
        deal arrived over two frames (1714, then 1266, each under the
        threshold) because the previous hand's cards were dimmed, and the
        old hand stayed up for 15 frames. If the two contradicting reads
        agree with each other, they are the new hand.
      * A board has appeared: then an "empty board" now is the run-it-once
        showdown rendering the board elsewhere, and the reads may be of
        cards in any state. The hand is withdrawn for the rest of it and
        never replaced, because replacing could install a wrong one.
    """

    def __init__(self):
        self.cards = []
        self.read_this_hand = False      # a hand has been confirmed and shown
        self._last_signature = None
        self._candidate = None
        self._streak = 0
        self._doubts = 0
        self._doubted = None
        self._withdrawn = False
        self._board_seen = False

    def _new_hand(self):
        self.cards = []
        self.read_this_hand = False
        self._candidate = None
        self._streak = 0
        self._doubts = 0
        self._doubted = None
        self._withdrawn = False
        self._board_seen = False

    def update(self, board, read, signature=None):
        """Return the hole cards to display.

        `read` is called at most once per frame and only at preflop; it returns
        a list of cards, or None/[] if none could be read. `signature` is a
        coarse thumbnail of the hero card region, or None when the payload
        carries no hero pixels -- in which case the board alone decides.
        """
        # Compared frame to frame, not against the signature at read time. A
        # deal is a sudden jump; the region also drifts slowly within a hand as
        # the bet slider and glow come and go, and measuring against the read
        # let that drift accumulate until it crossed the threshold and blanked
        # cards that were still perfectly valid.
        if self._last_signature is not None and _changed(signature, self._last_signature):
            self._new_hand()
        self._last_signature = signature

        if board and self.read_this_hand:
            self._board_seen = True

        if not board and not self._withdrawn:
            found = read() or []
            # A half-read or impossible pair says nothing either way: it is
            # neither agreement nor contradiction, and the deal animation
            # produces plenty of them.
            if _is_a_hand(found):
                if self.read_this_hand:
                    if found == self.cards:
                        self._doubts, self._doubted = 0, None
                    else:
                        same = found == self._doubted
                        self._doubts = self._doubts + 1 if same or self._doubts == 0 else 1
                        self._doubted = list(found)
                        if self._doubts >= AGREE:
                            if not self._board_seen and same:
                                self.cards = list(found)        # the missed deal
                                self._doubts, self._doubted = 0, None
                            else:
                                self.cards = []
                                self._withdrawn = True
                else:
                    if found == self._candidate:
                        self._streak += 1
                    else:
                        self._candidate, self._streak = list(found), 1
                    if self._streak >= AGREE:
                        self.cards = list(found)
                        self.read_this_hand = True

        # A card cannot be in the board and the hand at once. When both
        # readers claim one, one of them is wrong and nothing here can tell
        # which -- so report neither rather than present an impossible hand as
        # fact. Found by backtesting: a board ace of spades read as a diamond
        # and collided with the hero's real ace of diamonds.
        if self.cards and set(self.cards) & set(board):
            return []
        if self._doubts:
            return []
        return self.cards
