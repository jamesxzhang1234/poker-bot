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
    for i, (rank, bgr) in enumerate([("Q", (160, 60, 40)), ("7", (40, 40, 160))]):
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

    assert state.board_cards == ["Qd", "7h"]


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
