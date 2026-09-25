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


def _changed(a, b):
    if a is None or b is None or len(a) != len(b):
        return True
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b)) > CHANGE_THRESHOLD


class HeroCardTracker:
    def __init__(self):
        self.cards = []
        self.read_this_hand = False
        self._last_signature = None

    def update(self, board, read, signature=None):
        """Return the hole cards to display, reading them only when needed.

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
            self.cards = []
            self.read_this_hand = False
        self._last_signature = signature

        if not board and not self.read_this_hand:
            # Keep trying across preflop frames: the first can land mid-deal,
            # before the cards are legible, and giving up there would lose them
            # for the whole hand.
            found = read() or []
            if found:
                self.cards = list(found)
                self.read_this_hand = True

        # A card cannot be in the board and the hand at once. When both
        # readers claim one, one of them is wrong and nothing here can tell
        # which -- so report neither rather than present an impossible hand as
        # fact. Found by backtesting: a board ace of spades read as a diamond
        # and collided with the hero's real ace of diamonds.
        if self.cards and set(self.cards) & set(board):
            return []

        return self.cards
