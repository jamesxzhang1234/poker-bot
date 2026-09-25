"""Tests for turning Apple Vision text observations into table state.

The Swift capture helper runs one VNRecognizeTextRequest per frame and emits
a JSON payload of recognised strings with their pixel positions. These tests
drive the Python side that turns that payload into a TableState -- no fixed
per-field pixel boxes, text is assigned to seats by proximity to a seat
anchor, so the same code works on any layout given its anchors.
"""
import json

from poker_vision import vision_reader


def test_parses_observations_from_a_vision_payload():
    payload = json.dumps({
        "frame_index": 3,
        "timestamp_sec": 1.5,
        "width": 1640,
        "height": 2360,
        "observations": [
            {"text": "drake81", "x": 511, "y": 565, "w": 130, "h": 34, "confidence": 1.0},
            {"text": "7.75", "x": 509, "y": 620, "w": 80, "h": 30, "confidence": 1.0},
        ],
    })

    frame = vision_reader.parse_payload(payload)

    assert frame.frame_index == 3
    assert frame.timestamp_sec == 1.5
    assert frame.width == 1640
    assert frame.height == 2360
    assert [o.text for o in frame.observations] == ["drake81", "7.75"]
    assert frame.observations[0].x == 511
    assert frame.observations[0].y == 565
    assert frame.observations[1].confidence == 1.0


def test_parses_a_plain_decimal_amount():
    assert vision_reader.parse_amount("7.75") == 7.75
    assert vision_reader.parse_amount("20") == 20.0
    assert vision_reader.parse_amount("0.1") == 0.1


def test_rejects_an_amount_cut_off_mid_number():
    # Seen on real frames: a bet-sizing panel overlaps a seat and Vision reads
    # the visible part only ("19." of "19.45"). A trailing separator means we
    # are looking at occluded text, and guessing the rest would fabricate a
    # stack -- so refuse it rather than return 19.0.
    assert vision_reader.parse_amount("19.") is None
    assert vision_reader.parse_amount(".") is None


def test_rejects_strings_that_merely_contain_a_number():
    # "$0.05/$0.10" is the blind level, "100%" the battery, "2x" a raise
    # preset, "9:41AM" the clock -- none of them are amounts on the table.
    for text in ["$0.05/$0.10", "100%", "2x", "9:41AM", "+30s", "Total Pot"]:
        assert vision_reader.parse_amount(text) is None, text


def _obs(text, x, y, conf=1.0):
    return vision_reader.TextObservation(text=text, x=x, y=y, w=100, h=30, confidence=conf)


def test_pairs_a_name_with_the_amount_directly_beneath_it():
    # Every seat renders as a two-row plate: the handle, and the stack
    # immediately below it. That vertical stacking is what identifies a
    # seat plate, on either device's layout.
    obs = [_obs("drake81", 511, 565), _obs("7.75", 509, 620)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert len(plates) == 1
    assert plates[0].name == "drake81"
    assert plates[0].stack == 7.75
    assert plates[0].x == 511
    assert plates[0].y == 565


def test_does_not_pair_a_name_with_a_distant_amount():
    # The pot sits in the middle of the table, far from any nameplate.
    obs = [_obs("drake81", 511, 565), _obs("8.2", 818, 914)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert plates == []


def test_pairs_each_name_with_its_own_stack_when_several_seats_are_present():
    obs = [
        _obs("drake81", 511, 565), _obs("7.75", 509, 620),
        _obs("Jaimie73", 404, 895), _obs("9.45", 404, 951),
        _obs("trill04", 1195, 1721), _obs("9.4", 1193, 1778),
    ]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert {(p.name, p.stack) for p in plates} == {
        ("drake81", 7.75), ("Jaimie73", 9.45), ("trill04", 9.4),
    }


def test_does_not_treat_the_pot_label_as_a_seat_plate():
    # "Total Pot" sits directly above the pot amount with exactly the same
    # geometry as a handle above its stack, so pure geometry mistakes the pot
    # for a player. Real positions from frames/ipad/ipad_000.png.
    obs = [_obs("Total Pot", 818, 878), _obs("0.15", 807, 914)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert plates == []


def test_does_not_treat_a_status_word_as_a_handle():
    # A folded seat renders "Folded" above the plate; the action buttons along
    # the bottom ("Check", "Bet") and the table chrome ("NLH", "Invite") can
    # all land above some amount too.
    for label in ["Folded", "Check", "Bet", "NLH", "Invite", "PurePoker", "All In"]:
        obs = [_obs(label, 500, 800), _obs("9.45", 500, 850)]
        assert vision_reader.find_seat_plates(obs, width=1640, height=2360) == [], label


def test_reads_the_pot_from_the_amount_under_the_total_pot_label():
    obs = [
        _obs("Total Pot", 818, 878), _obs("0.15", 807, 914),
        _obs("drake81", 511, 565), _obs("7.75", 509, 620),
    ]

    assert vision_reader.find_pot(obs, width=1640, height=2360) == 0.15


def test_pot_is_none_when_the_label_is_absent():
    obs = [_obs("drake81", 511, 565), _obs("7.75", 509, 620)]

    assert vision_reader.find_pot(obs, width=1640, height=2360) is None


_ANCHORS = {
    "top_left": (511, 565),
    "mid_left": (404, 895),
    "lower_mid_left": (400, 1390),
    "lower_mid_right": (1230, 1390),
    "bottom_right": (1195, 1721),
    "hero": (814, 1920),
}


def test_assigns_each_plate_to_its_nearest_seat_anchor():
    plates = [
        vision_reader.SeatPlate("drake81", 7.75, 511, 565),
        vision_reader.SeatPlate("trill04", 9.4, 1193, 1725),
    ]

    assigned = vision_reader.assign_to_seats(plates, _ANCHORS, width=1640, height=2360)

    assert assigned["top_left"].name == "drake81"
    assert assigned["bottom_right"].name == "trill04"


def test_leaves_a_plate_unassigned_when_it_is_far_from_every_anchor():
    # Guards against a stray pairing somewhere in the table chrome being
    # snapped onto whichever seat happens to be least far away.
    plates = [vision_reader.SeatPlate("ghost", 1.0, 820, 200)]

    assigned = vision_reader.assign_to_seats(plates, _ANCHORS, width=1640, height=2360)

    assert assigned == {}


def test_two_plates_near_one_anchor_keep_only_the_closest():
    plates = [
        vision_reader.SeatPlate("real", 9.0, 511, 565),
        vision_reader.SeatPlate("stray", 1.0, 560, 600),
    ]

    assigned = vision_reader.assign_to_seats(plates, _ANCHORS, width=1640, height=2360)

    assert assigned["top_left"].name == "real"
    assert len(assigned) == 1


def test_ignores_observations_below_the_confidence_floor():
    # Vision occasionally reads card-back artwork as text -- on real iPad
    # frames it produced Cyrillic strings like "УрАмі" above a seat, which
    # then paired with that seat's stack and invented a player. Those come
    # back at low confidence while real table text scores 1.0.
    obs = [_obs("УрАмі", 516, 432, conf=0.3),
           _obs("7.75", 509, 480, conf=1.0)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert plates == []


def test_keeps_observations_at_or_above_the_confidence_floor():
    obs = [_obs("drake81", 511, 565, conf=0.9), _obs("7.75", 509, 620, conf=1.0)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert len(plates) == 1


def test_builds_a_table_state_from_a_vision_payload():
    import json
    from poker_vision import seat_anchors

    payload = json.dumps({
        "frame_index": 7, "timestamp_sec": 2.5, "width": 1640, "height": 2360,
        "observations": [
            {"text": "drake81", "x": 511, "y": 565, "w": 123, "h": 29, "confidence": 1.0},
            {"text": "7.75", "x": 509, "y": 620, "w": 72, "h": 34, "confidence": 1.0},
            {"text": "Jaimie73", "x": 404, "y": 895, "w": 130, "h": 30, "confidence": 1.0},
            {"text": "9.45", "x": 404, "y": 951, "w": 75, "h": 30, "confidence": 1.0},
            {"text": "Total Pot", "x": 818, "y": 878, "w": 106, "h": 20, "confidence": 1.0},
            {"text": "0.15", "x": 807, "y": 914, "w": 92, "h": 37, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    assert state.frame_index == 7
    assert state.timestamp_sec == 2.5
    assert state.pot == 0.15
    by_id = {s.seat_id: s for s in state.seats}
    assert by_id["top_left"].name == "drake81"
    assert by_id["top_left"].stack == 7.75
    assert by_id["mid_left"].name == "Jaimie73"
    assert by_id["mid_left"].stack == 9.45


def test_hero_seat_is_flagged():
    import json
    from poker_vision import seat_anchors

    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "jxz", "x": 814, "y": 1920, "w": 51, "h": 34, "confidence": 1.0},
            {"text": "27.8", "x": 820, "y": 1980, "w": 82, "h": 37, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    hero = [s for s in state.seats if s.is_hero]
    assert len(hero) == 1
    assert hero[0].name == "jxz"


def test_payload_carries_its_source_filename_when_present():
    # Directory mode names the file each payload came from. Consumers must key
    # on this rather than on line order: a frame that fails to decode is
    # skipped, so zipping payloads against a directory listing by position
    # pairs every later frame with the wrong image.
    import json
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1206, "height": 2622,
        "source": "IMG_8437.PNG", "observations": [],
    })

    assert vision_reader.parse_payload(payload).source == "IMG_8437.PNG"


def test_source_is_none_for_a_live_payload():
    import json
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [],
    })

    assert vision_reader.parse_payload(payload).source is None


def test_assigns_a_persistent_status_label_to_the_seat_below_it():
    # "Folded" renders just above the plate: 65-69px on the iPad's 2360-high
    # frame, 70-76px on the iPhone's 2622 -- 2.7-2.9% of frame height on both.
    obs = [_obs("Folded", 404, 828), _obs("Jaimie73", 404, 895), _obs("9.45", 404, 951)]

    statuses = vision_reader.find_statuses(obs, width=1640, height=2360)

    assert statuses == {("Jaimie73", 404, 895): "Folded"}


def test_assigns_an_action_bubble_to_the_seat_below_it():
    # The action bubble floats higher than the persistent label -- ~216px on
    # the iPad, ~240px on the iPhone, i.e. 9.1-9.3% of height on both.
    obs = [_obs("Call", 440, 679), _obs("Jaimie73", 404, 895), _obs("9.45", 404, 951)]

    statuses = vision_reader.find_statuses(obs, width=1640, height=2360)

    assert statuses == {("Jaimie73", 404, 895): "Call"}


def test_does_not_assign_the_bottom_action_buttons_to_a_seat():
    # "Fold"/"Check"/"Bet" also label the hero's buttons along the bottom of
    # the screen. Those sit BELOW every plate, so requiring the status to be
    # above its seat is what keeps them out.
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980), _obs("Fold", 289, 2258)]

    assert vision_reader.find_statuses(obs, width=1640, height=2360) == {}


def test_does_not_reach_across_to_a_horizontally_distant_seat():
    # A status on the left must not attach to a plate on the right that
    # happens to sit at the right height.
    obs = [_obs("Folded", 401, 828), _obs("kidpokers", 1233, 897), _obs("20", 1231, 951)]

    assert vision_reader.find_statuses(obs, width=1640, height=2360) == {}


def test_table_state_carries_seat_status():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "Folded", "x": 404, "y": 828, "w": 90, "h": 28, "confidence": 1.0},
            {"text": "Jaimie73", "x": 404, "y": 895, "w": 130, "h": 30, "confidence": 1.0},
            {"text": "9.45", "x": 404, "y": 951, "w": 75, "h": 30, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    seat = [s for s in state.seats if s.seat_id == "mid_left"][0]
    assert seat.status == "Folded"
    assert seat.in_hand is False


def test_a_seat_with_an_action_bubble_is_still_in_the_hand():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "Call", "x": 440, "y": 679, "w": 70, "h": 28, "confidence": 1.0},
            {"text": "Jaimie73", "x": 404, "y": 895, "w": 130, "h": 30, "confidence": 1.0},
            {"text": "9.45", "x": 404, "y": 951, "w": 75, "h": 30, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    seat = [s for s in state.seats if s.seat_id == "mid_left"][0]
    assert seat.status == "Call"
    assert seat.in_hand is True


def test_status_is_normalised_to_a_canonical_spelling():
    # Vision returns whatever casing it read -- on real frames it produced
    # both "Folded" and "folded". A consumer comparing status == "Folded"
    # would silently miss the second, so the reader settles on one spelling.
    obs = [_obs("folded", 404, 828), _obs("Jaimie73", 404, 895), _obs("9.45", 404, 951)]

    statuses = vision_reader.find_statuses(obs, width=1640, height=2360)

    assert statuses == {("Jaimie73", 404, 895): "Folded"}


def _png_b64(img):
    import base64, cv2
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def test_payload_carries_a_decoded_board_strip():
    # The helper sends recognised text, not pixels -- but card ranks are
    # template-matched, which needs pixels. Rather than ship a 4MB frame it
    # attaches just the board band as a small PNG.
    import json
    import numpy as np
    strip = np.full((120, 600, 3), 40, dtype=np.uint8)
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "board_strip": _png_b64(strip),
        "board_strip_rect": [0.18, 0.40, 0.63, 0.10],
        "observations": [],
    })

    frame = vision_reader.parse_payload(payload)

    assert frame.board_strip is not None
    assert frame.board_strip.shape == (120, 600, 3)
    assert frame.board_strip_rect == (0.18, 0.40, 0.63, 0.10)


def test_board_strip_is_none_when_the_helper_sends_no_pixels():
    import json
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [],
    })

    frame = vision_reader.parse_payload(payload)

    assert frame.board_strip is None
    assert frame.board_strip_rect is None


def test_translates_frame_slots_into_strip_coordinates():
    # The strip is a crop, so a slot's frame coordinates have to be shifted by
    # the crop's origin and rescaled by how much the strip was resized.
    slots = [(495, 985, 613, 1139)]
    rect = (0.18, 0.40, 0.63, 0.10)          # x, y, w, h as fractions

    moved = vision_reader.slots_in_strip(
        slots, rect, frame_size=(1640, 2360), strip_size=(1033, 236))

    # crop origin is (0.18*1640, 0.40*2360) = (295.2, 944); no rescale here
    assert moved == [(200, 41, 318, 195)]


def test_slot_translation_accounts_for_a_rescaled_strip():
    slots = [(495, 985, 613, 1139)]
    rect = (0.18, 0.40, 0.63, 0.10)

    moved = vision_reader.slots_in_strip(
        slots, rect, frame_size=(1640, 2360), strip_size=(516, 118))   # half size

    assert moved == [(100, 20, 159, 98)]


def test_table_state_reads_the_board_from_the_attached_strip():
    import json
    import cv2
    import numpy as np
    from poker_vision import rank_templates, seat_anchors

    # Build a frame-sized canvas, draw two cards into the real iPad board
    # slots, then hand over only the cropped strip -- exactly what the helper
    # sends.
    frame = np.full((2360, 1640, 3), 30, dtype=np.uint8)
    templates = rank_templates.load()
    # three cards, not two: a board is only ever 0, 3, 4 or 5 long
    for i, (rank, bgr) in enumerate([("Q", (160, 60, 40)), ("7", (40, 40, 160)),
                                     ("9", (50, 150, 50))]):
        x1, y1, x2, y2 = seat_anchors.IPAD_BOARD_SLOTS[i]
        frame[y1:y2, x1:x2] = bgr
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 110), 30, (255, 255, 255), -1)
        gx1, gy1, gx2, gy2 = seat_anchors.IPAD_RANK_OFFSET
        glyph = cv2.resize(templates[rank][0], (gx2 - gx1, gy2 - gy1),
                           interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + gy1:y1 + gy2, x1 + gx1:x1 + gx2]
        region[glyph > 127] = (255, 255, 255)

    rect = (0.18, 0.40, 0.63, 0.10)
    sx, sy = int(rect[0] * 1640), int(rect[1] * 2360)
    sw, sh = int(rect[2] * 1640), int(rect[3] * 2360)
    strip = frame[sy:sy + sh, sx:sx + sw]

    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "board_strip": _png_b64(strip), "board_strip_rect": list(rect),
        "observations": [],
    })

    state = vision_reader.build_table_state(
        payload, seat_anchors.IPAD_PORTRAIT,
        board_slots=seat_anchors.IPAD_BOARD_SLOTS,
        rank_offset=seat_anchors.IPAD_RANK_OFFSET)

    assert state.board_cards == ["Qd", "7h", "9c"]


def test_board_is_empty_when_the_helper_sent_no_strip():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [],
    })

    state = vision_reader.build_table_state(
        payload, seat_anchors.IPAD_PORTRAIT,
        board_slots=seat_anchors.IPAD_BOARD_SLOTS,
        rank_offset=seat_anchors.IPAD_RANK_OFFSET)

    assert state.board_cards == []


def test_assigns_a_position_badge_to_its_seat():
    # Badges sit above and to one side of the plate, at 8.7-9.8% of the frame
    # diagonal on both devices. Real coordinates from frames/ipad.
    obs = [_obs("SB", 924, 1689), _obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]

    badges = vision_reader.find_badges(obs, width=1640, height=2360)

    assert badges == {("jxz", 814, 1920): "SB"}


def test_a_badge_below_a_plate_belongs_to_a_different_seat():
    # Every badge renders above its own seat, so one sitting below a plate is
    # another seat's and must not be attached to this one.
    obs = [_obs("BB", 814, 2100), _obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]

    assert vision_reader.find_badges(obs, width=1640, height=2360) == {}


def test_a_distant_badge_is_not_attached():
    obs = [_obs("SB", 200, 400), _obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]

    assert vision_reader.find_badges(obs, width=1640, height=2360) == {}


def test_table_state_carries_the_badge_position():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "SB", "x": 924, "y": 1689, "w": 37, "h": 24, "confidence": 1.0},
            {"text": "jxz", "x": 814, "y": 1920, "w": 51, "h": 34, "confidence": 1.0},
            {"text": "27.8", "x": 820, "y": 1980, "w": 82, "h": 37, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    hero = [s for s in state.seats if s.seat_id == "hero"][0]
    assert hero.position == "SB"


def test_unread_positions_are_inferred_from_one_detected_badge():
    # Vision never returns a bare "D" -- a single character is too isolated to
    # be detected as text -- but it does read "SB"/"BB". One badge is enough
    # for parser.infer_positions to rotate the rest of the table into place.
    import json
    from poker_vision import seat_anchors

    seats = [("drake81", 511, 565, "7.75"), ("Jaimie73", 404, 895, "9.45"),
             ("edngaming", 393, 1392, "20.8"), ("trill04", 1195, 1721, "9.4"),
             ("jxz", 814, 1920, "27.8")]
    obs = [{"text": "SB", "x": 924, "y": 1689, "w": 37, "h": 24, "confidence": 1.0}]
    for name, x, y, stack in seats:
        obs.append({"text": name, "x": x, "y": y, "w": 120, "h": 30, "confidence": 1.0})
        obs.append({"text": stack, "x": x, "y": y + 57, "w": 70, "h": 30, "confidence": 1.0})
    payload = json.dumps({"frame_index": 0, "timestamp_sec": 0.0,
                          "width": 1640, "height": 2360, "observations": obs})

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    positions = {s.seat_id: s.position for s in state.seats}
    assert positions["hero"] == "SB"
    # every occupied seat ends up with a position, not just the badged one
    assert all(p is not None for p in positions.values()), positions


def test_a_seat_whose_stack_is_occluded_is_still_reported():
    # A bet-sizing panel over a seat leaves the handle readable and the stack
    # cut off ("19." of "19.45"). Dropping the plate loses the player
    # entirely; reporting them with an unknown stack says what is actually
    # known. Real coordinates from frames/ipad/ipad_015.png.
    obs = [_obs("trill04", 1183, 1721), _obs("19.", 1188, 1780)]

    plates = vision_reader.find_seat_plates(obs, width=1640, height=2360)

    assert len(plates) == 1
    assert plates[0].name == "trill04"
    assert plates[0].stack is None


def test_a_handle_with_nothing_beneath_it_is_not_a_seat():
    # Distinguishes an occluded stack from a stray word that merely sits
    # where a handle would.
    obs = [_obs("trill04", 1183, 1721)]

    assert vision_reader.find_seat_plates(obs, width=1640, height=2360) == []


def test_a_handle_over_non_numeric_text_is_not_a_seat():
    obs = [_obs("trill04", 1183, 1721), _obs("Straddle", 1188, 1780)]

    assert vision_reader.find_seat_plates(obs, width=1640, height=2360) == []


def test_an_occluded_seat_reaches_the_table_state_with_no_stack():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "trill04", "x": 1183, "y": 1721, "w": 93, "h": 28, "confidence": 1.0},
            {"text": "19.", "x": 1188, "y": 1780, "w": 54, "h": 30, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    seat = [s for s in state.seats if s.seat_id == "bottom_right"][0]
    assert seat.name == "trill04"
    assert seat.stack is None


def test_a_badge_read_with_a_lookalike_character_still_counts():
    # Vision returned "вB" on a real iPhone frame -- a Cyrillic ve standing in
    # for the first B. One badge is all the position inference needs, so
    # losing it to a single wrong codepoint costs the whole table's positions.
    obs = [_obs("вB", 924, 1689), _obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]

    badges = vision_reader.find_badges(obs, width=1640, height=2360)

    assert badges == {("jxz", 814, 1920): "BB"}


def test_lookalike_folding_does_not_invent_badges_from_other_text():
    for text in ["8", "5", "0", "13", "6.3", "BET"]:
        obs = [_obs(text, 924, 1689), _obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]
        assert vision_reader.find_badges(obs, width=1640, height=2360) == {}, text


def test_reports_plates_that_match_no_seat_anchor():
    # A plate landing nowhere near a known anchor means the layout has a seat
    # this device's anchor set doesn't describe -- iPad bottom_left is a
    # mirrored guess nobody has sat in yet. Silently dropping it hides that;
    # surfacing it turns a missing player into a visible gap.
    plates = [
        vision_reader.SeatPlate("drake81", 7.75, 511, 565),
        vision_reader.SeatPlate("mystery", 3.0, 445, 1721),
    ]
    anchors = {"top_left": (511, 565)}

    orphans = vision_reader.plates_without_seat(plates, anchors, width=1640, height=2360)

    assert [p.name for p in orphans] == ["mystery"]


def test_no_orphans_when_every_plate_lands_on_an_anchor():
    plates = [vision_reader.SeatPlate("drake81", 7.75, 511, 565)]
    anchors = {"top_left": (511, 565)}

    assert vision_reader.plates_without_seat(plates, anchors, 1640, 2360) == []


def test_reads_an_amount_written_with_letter_lookalikes():
    # Vision returned "IO" for a stack of 10 on a real frame. Capital I and
    # lowercase l are the same shape as 1 in this UI's font, and O is the same
    # shape as 0, so a single substitution loses the whole seat's stack.
    assert vision_reader.parse_amount("IO") == 10.0
    assert vision_reader.parse_amount("l0") == 10.0
    assert vision_reader.parse_amount("2O.5") == 20.5


def test_letter_folding_does_not_turn_words_into_amounts():
    # Folding must not manufacture a number out of ordinary text.
    for text in ["Invite", "All In", "Fold", "IOU", "LOL", "Total Pot", "NLH"]:
        assert vision_reader.parse_amount(text) is None, text


def test_payload_carries_a_decoded_text_mask():
    import json
    import cv2
    import numpy as np
    mask = np.zeros((200, 100), dtype=np.uint8)
    mask[50:60, 20:40] = 255
    import base64
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "text_mask": base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode(),
        "observations": [],
    })

    frame = vision_reader.parse_payload(payload)

    assert frame.text_mask is not None
    assert frame.text_mask.shape == (200, 100)


def test_text_mask_is_none_when_absent():
    import json
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [],
    })

    assert vision_reader.parse_payload(payload).text_mask is None


def test_recovers_a_short_stack_that_vision_missed():
    # Vision reads the handle but not a one-character stack beneath it -- 19
    # of the 29 reads the Tesseract parser found and this path did not were
    # single characters. With the frame's text mask attached, that one seat's
    # stack row can be re-read rather than left unknown.
    import base64
    import json
    import cv2
    import numpy as np
    from poker_vision import seat_anchors

    mask = np.zeros((2360, 1640), dtype=np.uint8)
    # a "5" drawn where mid_left's stack row sits, at a realistic glyph size
    cv2.putText(mask, "5", (380, 975), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)

    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "text_mask": base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode(),
        "observations": [
            {"text": "Jaimie73", "x": 404, "y": 895, "w": 130, "h": 30, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    seat = [s for s in state.seats if s.seat_id == "mid_left"][0]
    assert seat.name == "Jaimie73"
    assert seat.stack == 5.0


def test_a_crop_with_no_digit_sized_marks_is_rejected_without_ocr():
    # Recovery runs Tesseract, which costs ~56ms a call, and it is attempted
    # for every seat missing a stack -- most of which have nothing there. A
    # cheap shape check keeps the expensive call for crops that could actually
    # hold digits.
    import numpy as np
    speckle = np.zeros((100, 250), dtype=np.uint8)
    speckle[10:13, 20:23] = 255          # far too small to be a glyph
    speckle[60:62, 200:203] = 255

    assert vision_reader.looks_like_digits(speckle, height=2360) is False


def test_a_crop_with_a_digit_sized_mark_is_accepted():
    import cv2
    import numpy as np
    crop = np.zeros((100, 250), dtype=np.uint8)
    cv2.putText(crop, "5", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)

    assert vision_reader.looks_like_digits(crop, height=2360) is True


def test_an_empty_crop_is_rejected():
    import numpy as np
    assert vision_reader.looks_like_digits(np.zeros((100, 250), dtype=np.uint8),
                                           height=2360) is False


def test_attaches_a_bet_to_the_seat_that_made_it():
    # Chips a player has committed this street render on the felt between
    # their seat and the pot. Real coordinates from frames/ipad/ipad_015.png:
    # hero's 0.05 sits 267px from the hero anchor.
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980), _obs("0.05", 816, 1653)]

    bets = vision_reader.find_bets(obs, vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors_ipad(), 1640, 2360), 1640, 2360)

    assert bets == {"hero": 0.05}


def seat_anchors_ipad():
    from poker_vision import seat_anchors
    return seat_anchors.IPAD_PORTRAIT


def test_the_small_number_above_a_plate_is_not_a_bet():
    # Each seat carries a small count just above its plate (85-98px away).
    # Bets sit much further out, 165-270px, so the near ones must be excluded
    # or every seat would report a bet it never made.
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980), _obs("34", 892, 1860)]

    bets = vision_reader.find_bets(obs, vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors_ipad(), 1640, 2360), 1640, 2360)

    assert bets == {}


def test_the_pot_is_not_attached_as_a_bet():
    # The pot sits ~400px from the nearest seat, well beyond any bet.
    obs = [_obs("Jaimie73", 404, 895), _obs("9.45", 404, 951),
           _obs("Total Pot", 818, 878), _obs("0.55", 809, 914)]

    bets = vision_reader.find_bets(obs, vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors_ipad(), 1640, 2360), 1640, 2360)

    assert bets == {}


def test_table_state_carries_each_seat_bet():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "jxz", "x": 814, "y": 1920, "w": 51, "h": 34, "confidence": 1.0},
            {"text": "27.8", "x": 820, "y": 1980, "w": 82, "h": 37, "confidence": 1.0},
            {"text": "0.05", "x": 816, "y": 1653, "w": 79, "h": 34, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    hero = [s for s in state.seats if s.seat_id == "hero"][0]
    assert hero.bet == 0.05


def test_a_board_card_rank_is_not_read_as_a_bet():
    # The board sits in the middle of the table, inside the distance band
    # where bets live. On a real frame the 5 of spades was attributed to
    # mid_left as a 5.0 bet, which then made the bets disagree with the pot.
    from poker_vision import seat_anchors
    obs = [_obs("Jaimie73", 404, 895), _obs("9.45", 404, 951),
           _obs("5", 524, 1013)]          # leftmost board card's rank
    assigned = vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors.IPAD_PORTRAIT, 1640, 2360)

    bets = vision_reader.find_bets(obs, assigned, 1640, 2360,
                                   board_slots=seat_anchors.IPAD_BOARD_SLOTS)

    assert bets == {}


def test_a_real_bet_outside_the_board_is_still_read():
    from poker_vision import seat_anchors
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980), _obs("0.05", 816, 1653)]
    assigned = vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors.IPAD_PORTRAIT, 1640, 2360)

    bets = vision_reader.find_bets(obs, assigned, 1640, 2360,
                                   board_slots=seat_anchors.IPAD_BOARD_SLOTS)

    assert bets == {"hero": 0.05}


def test_a_hero_hole_card_rank_is_not_read_as_a_bet():
    # The hero's own cards render beside their seat, inside the bet distance
    # band. On a real frame the 9 of spades became a 9.0 bet for hero, which
    # is larger than the entire pot.
    from poker_vision import seat_anchors
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980), _obs("9", 977, 1866)]
    assigned = vision_reader.assign_to_seats(
        vision_reader.find_seat_plates(obs, 1640, 2360),
        seat_anchors.IPAD_PORTRAIT, 1640, 2360)

    bets = vision_reader.find_bets(
        obs, assigned, 1640, 2360,
        board_slots=seat_anchors.IPAD_BOARD_SLOTS,
        hero_cards=seat_anchors.IPAD_HERO_CARDS)

    assert bets == {}


def test_payload_carries_a_decoded_hero_strip():
    import base64, json
    import cv2
    import numpy as np
    strip = np.full((300, 400, 3), 30, dtype=np.uint8)
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "hero_strip": base64.b64encode(cv2.imencode(".png", strip)[1].tobytes()).decode(),
        "hero_strip_rect": [0.55, 0.74, 0.30, 0.15],
        "observations": [],
    })

    frame = vision_reader.parse_payload(payload)

    assert frame.hero_strip is not None
    assert frame.hero_strip.shape == (300, 400, 3)
    assert frame.hero_strip_rect == (0.55, 0.74, 0.30, 0.15)


def test_reading_hero_cards_with_no_strip_returns_nothing():
    import json
    from poker_vision import seat_anchors
    frame = vision_reader.parse_payload(json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": []}))

    assert vision_reader.read_hero_cards(
        frame, seat_anchors.IPAD_HERO_CARD_SLOTS) == []


def test_reads_hero_cards_out_of_the_strip():
    import base64, json
    import cv2
    import numpy as np
    from poker_vision import rank_templates, seat_anchors

    # Draw two cards into the real hero slots on a frame-sized canvas, then
    # hand over only the cropped strip -- what the helper actually sends.
    # A real card is a coloured face with a border, a rank glyph and a large
    # suit pip. is_card_present keys on edge density, and a card drawn with
    # just a letter on a flat rectangle falls under the threshold -- the
    # fixture has to look like a card, not merely be one.
    frame_img = np.full((2360, 1640, 3), 25, dtype=np.uint8)
    for (x1, y1, x2, y2), (rank, bgr) in zip(
            seat_anchors.IPAD_HERO_CARD_SLOTS,
            [("A", (160, 60, 40)), ("K", (40, 40, 160))]):
        frame_img[y1:y2, x1:x2] = bgr
        # Real harvested glyphs, not a drawing font: OpenCV's Hershey "K"
        # reads as a 4, which is a property of that font and nothing to do
        # with the code under test.
        glyph = cv2.resize(rank_templates.load()[rank][0], (52, 66),
                           interpolation=cv2.INTER_NEAREST)
        target = frame_img[y1 + 14:y1 + 80, x1 + 16:x1 + 68]
        target[glyph > 127] = (255, 255, 255)
        # The pip must stay clear of the left band sample_face_color averages.
        # Drawn oversized it reached into that band and washed the blue toward
        # white, turning the diamond into a spade -- a fixture artefact; real
        # pips do not sit that far left on a fanned card.
        cv2.circle(frame_img, ((x1 + x2) // 2 + 12, y2 - 48), 28,
                   (255, 255, 255), -1)
        # Detail confined to the bottom of the card. It has to be below the
        # rank, because read_rank_from_top crops the upper part and a border
        # there reads as part of the glyph (it turned the A into a T). It also
        # has to be below the middle, because sample_face_color averages the
        # left-middle band and white bleeding into that washes the suit colour
        # toward grey (it turned the diamond into a spade).
        cv2.rectangle(frame_img, (x1 + 4, y1 + int((y2 - y1) * 0.80)),
                      (x2 - 5, y2 - 4), (245, 245, 245), 3)

    rect = (0.55, 0.74, 0.30, 0.15)
    sx, sy = int(rect[0] * 1640), int(rect[1] * 2360)
    strip = frame_img[sy:sy + int(rect[3] * 2360), sx:sx + int(rect[2] * 1640)]
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "hero_strip": base64.b64encode(cv2.imencode(".png", strip)[1].tobytes()).decode(),
        "hero_strip_rect": list(rect), "observations": [],
    })

    cards = vision_reader.read_hero_cards(
        vision_reader.parse_payload(payload), seat_anchors.IPAD_HERO_CARD_SLOTS)

    assert cards == ["Ad", "Kh"]


def test_a_seat_holding_cards_is_in_the_hand():
    # A folded player has no cards beside their avatar; an active one has two
    # face-down backs up and left of their plate. That is a visual yes/no,
    # unlike the small red "Folded" text which is read on only 28% of seats.
    import numpy as np
    from poker_vision import seat_anchors
    thumb = np.zeros((590, 410), dtype=np.uint8)      # quarter-scale frame
    ax, ay = seat_anchors.IPAD_PORTRAIT["top_left"]
    x1, y1 = int((ax - 140) * 0.25), int((ay - 100) * 0.25)
    x2, y2 = int((ax - 20) * 0.25), int((ay - 20) * 0.25)
    thumb[y1:y2, x1:x2] = 40                          # dark card body
    thumb[y1 + 4:y1 + 12, x1 + 3:x2 - 3] = 190        # the light logo on the backs

    assert vision_reader.seat_has_cards(thumb, (ax, ay), 0.25) is True


def test_a_folded_seat_is_not_in_the_hand():
    import numpy as np
    from poker_vision import seat_anchors
    thumb = np.zeros((590, 410), dtype=np.uint8)
    ax, ay = seat_anchors.IPAD_PORTRAIT["top_left"]

    assert vision_reader.seat_has_cards(thumb, (ax, ay), 0.25) is False


def test_no_thumbnail_means_unknown_rather_than_folded():
    from poker_vision import seat_anchors
    assert vision_reader.seat_has_cards(None, seat_anchors.IPAD_PORTRAIT["top_left"],
                                        0.25) is None


def test_it_is_your_turn_when_the_real_action_buttons_show():
    obs = [_obs("Fold", 289, 2258), _obs("Check", 818, 2260), _obs("Bet", 1348, 2258)]

    action = vision_reader.find_action(obs, width=1640, height=2360)

    assert action.your_turn is True
    assert action.to_call is None


def test_the_call_button_carries_what_you_owe():
    obs = [_obs("Fold", 289, 2258), _obs("Call 0.25", 741, 2439)]

    action = vision_reader.find_action(obs, width=1640, height=2360)

    assert action.your_turn is True
    assert action.to_call == 0.25


def test_preaction_checkboxes_do_not_mean_it_is_your_turn():
    # "Call any" / "Check / Fold" / "Straddle" are the pre-select boxes shown
    # while WAITING. Vision sometimes splits "Check / Fold" into separate
    # Check and Fold tokens, which look exactly like the real buttons -- so a
    # pre-action word anywhere on the row settles it.
    obs = [_obs("Check / Fold", 289, 2257), _obs("Call any", 823, 2258),
           _obs("Straddle", 1346, 2257)]

    assert vision_reader.find_action(obs, width=1640, height=2360).your_turn is False

    split = [_obs("Check", 250, 2257), _obs("Fold", 330, 2257),
             _obs("Call any", 823, 2258)]

    assert vision_reader.find_action(split, width=1640, height=2360).your_turn is False


def test_no_buttons_at_all_means_not_your_turn():
    obs = [_obs("jxz", 814, 1920), _obs("27.8", 820, 1980)]

    action = vision_reader.find_action(obs, width=1640, height=2360)

    assert action.your_turn is False
    assert action.to_call is None


def test_table_state_carries_whether_it_is_your_turn_and_the_call():
    import json
    from poker_vision import seat_anchors
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "jxz", "x": 814, "y": 1920, "w": 51, "h": 34, "confidence": 1.0},
            {"text": "27.8", "x": 820, "y": 1980, "w": 82, "h": 37, "confidence": 1.0},
            {"text": "Fold", "x": 289, "y": 2258, "w": 90, "h": 38, "confidence": 1.0},
            {"text": "Call 0.3", "x": 823, "y": 2258, "w": 120, "h": 38, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    assert state.your_turn is True
    assert state.to_call == 0.3


def test_in_hand_comes_from_the_cards_beside_a_seat_when_available():
    import base64, json
    import cv2
    import numpy as np
    from poker_vision import seat_anchors

    thumb = np.zeros((590, 410), dtype=np.uint8)
    ax, ay = seat_anchors.IPAD_PORTRAIT["top_left"]
    x1, y1 = int((ax - 140) * 0.25), int((ay - 100) * 0.25)
    x2, y2 = int((ax - 20) * 0.25), int((ay - 20) * 0.25)
    thumb[y1:y2, x1:x2] = 40
    thumb[y1 + 4:y1 + 12, x1 + 3:x2 - 3] = 190

    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "table_thumb": base64.b64encode(cv2.imencode(".jpg", thumb)[1].tobytes()).decode(),
        "table_thumb_scale": 0.25,
        "observations": [
            {"text": "drake81", "x": 511, "y": 565, "w": 123, "h": 29, "confidence": 1.0},
            {"text": "7.75", "x": 509, "y": 620, "w": 72, "h": 34, "confidence": 1.0},
        ],
    })

    state = vision_reader.build_table_state(payload, seat_anchors.IPAD_PORTRAIT)

    seat = [s for s in state.seats if s.seat_id == "top_left"][0]
    assert seat.in_hand is True


def test_a_truncated_status_label_is_not_a_player():
    # Vision returned 'Sitting (100' on a real frame -- a "Sitting Out" label
    # cut short -- and it became a phantom player at the table. Matching the
    # full phrase is not enough when the phrase can arrive clipped.
    for text in ["Sitting (100", "Sitting Out (100 bb)", "sitting out", "Waiting for BB"]:
        obs = [_obs(text, 404, 895), _obs("9.45", 404, 951)]
        assert vision_reader.find_seat_plates(obs, 1640, 2360) == [], text


def test_a_handle_that_merely_starts_with_a_letter_is_kept():
    obs = [_obs("Sitty", 404, 895), _obs("9.45", 404, 951)]

    assert len(vision_reader.find_seat_plates(obs, 1640, 2360)) == 1
