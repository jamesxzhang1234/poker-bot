"""Holds each seat's position for the length of a hand.

The client draws the SB/BB badges while the blinds are posted and takes them
away once the betting moves on. Measured across a 241-frame live corpus, 93 of
143 pre-flop frames carry a badge against 19 of 98 post-flop ones, and the
dealer button is never readable at all -- it is a graphic, not text, so Vision
returns no token for it. position_inference needs one badge to anchor its
rotation, so on 53% of frames it had nothing to work from and every seat came
back with no position.

Nothing can be read off a screen that stops drawing it. But positions do not
change within a hand: the ones worked out while the badges were up are still
true after they go. So they are held, and thrown away at the next deal.

This is the second place in the pipeline that remembers anything between
frames, which CLAUDE.md otherwise rules out, and it earns the exception the
same way hero_cards.HeroCardTracker does -- by reasoning only about whether a
value it already read is still valid, never inferring one. It re-presents what
position_inference worked out earlier in this same hand and invents nothing:
a seat that never had a position never gets one, and a seat that has emptied
loses the one it had.

A stale position is worse than no position, because it looks like an answer.
So the hand boundary is the hero-card signature -- the same signal, and the
same threshold, that tells HeroCardTracker a new pair has been dealt -- and
where there is no signature to watch, nothing is held at all.
"""
from .hero_cards import _changed

def _is_occupied(seat):
    # Matches position_inference._is_occupied: a real player shows at least a
    # name or a stack. Deliberately not in_hand, which comes from a card-
    # presence heuristic that can fire on an empty seat's artwork.
    return bool(seat.name) or seat.stack is not None


class PositionTracker:
    def __init__(self):
        self._held = {}
        self._last_signature = None

    def update(self, seats, signature=None):
        """Fill in positions this frame could not read, from earlier this hand.

        `signature` is the coarse hero-card-region thumbnail that
        HeroCardTracker uses to spot a deal, or None when the payload carries
        no hero pixels.
        """
        if signature is None:
            # No way to tell one hand from the next. Anything held would
            # eventually be applied to the wrong players, so hold nothing.
            self._held = {}
            self._last_signature = None
            return seats

        if self._last_signature is not None and _changed(signature, self._last_signature):
            self._held = {}
        self._last_signature = signature

        read_now = {s.seat_id: s.position for s in seats if s.position}
        if read_now:
            # A badge read in this frame is better evidence than anything
            # held, so the whole mapping is replaced rather than merged --
            # merging could leave a role on two seats at once.
            self._held = dict(read_now)
            return seats

        for seat in seats:
            if seat.position is None and _is_occupied(seat):
                held = self._held.get(seat.seat_id)
                if held is not None:
                    seat.position = held
        return seats
