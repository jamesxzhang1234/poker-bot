import queue

import cv2
import numpy as np

from poker_vision.server import create_app, worker_loop


def test_upload_writes_file_and_enqueues_path(tmp_path):
    work_queue = queue.Queue()
    app = create_app(tmp_path / "incoming", work_queue)
    client = app.test_client()

    resp = client.post("/upload", data=b"fake-image-bytes", content_type="image/jpeg")

    assert resp.status_code == 200
    files = list((tmp_path / "incoming").glob("*.jpg"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"fake-image-bytes"
    assert work_queue.get_nowait() == str(files[0])


def test_upload_rejects_empty_body(tmp_path):
    work_queue = queue.Queue()
    app = create_app(tmp_path / "incoming", work_queue)
    client = app.test_client()

    resp = client.post("/upload", data=b"", content_type="image/jpeg")

    assert resp.status_code == 400
    assert work_queue.empty()


def test_worker_loop_logs_parse_error_for_unreadable_image(tmp_path, capsys):
    bad_path = tmp_path / "not-an-image.jpg"
    bad_path.write_bytes(b"not a real jpeg")
    work_queue = queue.Queue()
    work_queue.put(str(bad_path))
    work_queue.put(None)  # sentinel: stop the loop

    worker_loop(work_queue)

    out = capsys.readouterr().out
    assert "[parse error] unreadable image" in out


class _FakeState:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class _FakeParser:
    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        return _FakeState({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "pot": 10.0,
            "board_cards": [],
            "seats": [{"stack": 5.0} for _ in range(4)],
        })


def test_worker_loop_prints_summary_for_valid_image(tmp_path, capsys):
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    work_queue = queue.Queue()
    work_queue.put(str(img_path))
    work_queue.put(None)

    worker_loop(work_queue, parser_factory=_FakeParser)

    out = capsys.readouterr().out
    assert "Pot: 10.0" in out


class _ImplausibleParser:
    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        return _FakeState({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "pot": None,
            "board_cards": [],
            "seats": [{"stack": None} for _ in range(9)],
        })


def test_worker_loop_skips_implausible_table(tmp_path, capsys):
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    work_queue = queue.Queue()
    work_queue.put(str(img_path))
    work_queue.put(None)

    worker_loop(work_queue, parser_factory=_ImplausibleParser)

    out = capsys.readouterr().out
    assert "[skipped]" in out
    assert "Pot:" not in out


class _RaisingParser:
    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        raise ValueError("boom")


def test_worker_loop_logs_parse_error_when_parser_raises(tmp_path, capsys):
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    work_queue = queue.Queue()
    work_queue.put(str(img_path))
    work_queue.put(None)

    worker_loop(work_queue, parser_factory=_RaisingParser)

    out = capsys.readouterr().out
    assert "[parse error] boom" in out
