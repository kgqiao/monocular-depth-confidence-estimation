import numpy as np
import pytest

from confhead.nuscenes import RADAR_PCD_DTYPE, parse_radar_pcd

FIELDS = "x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms"
SIZE = "4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1"
TYPE = "F F F I I F F F F F I I I I I I I I"
COUNT = " ".join(["1"] * 18)


def pcd(points: np.ndarray, size: str = SIZE) -> bytes:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {FIELDS}\nSIZE {size}\nTYPE {TYPE}\nCOUNT {COUNT}\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    return header.encode() + points.tobytes() + b"\x00" * 7  # trailing bytes, as in nuScenes files


def records(n: int) -> np.ndarray:
    rec = np.zeros(n, dtype=RADAR_PCD_DTYPE)
    rec["x"] = np.arange(n) + 10.0
    rec["ambig_state"] = 3
    return rec


def test_parses_the_records_and_ignores_trailing_bytes():
    rec = parse_radar_pcd(pcd(records(3)), "a.pcd")
    assert len(rec) == 3
    assert rec["x"].tolist() == [10.0, 11.0, 12.0]
    assert (rec["ambig_state"] == 3).all()


def test_a_header_that_does_not_match_the_layout_fails_with_the_file_name():
    with pytest.raises(ValueError, match=r"b\.pcd: PCD SIZE"):
        parse_radar_pcd(pcd(records(1), size=SIZE.replace("2", "4", 1)), "b.pcd")


def test_ascii_or_compressed_data_fails():
    with pytest.raises(ValueError, match="not a binary PCD"):
        parse_radar_pcd(pcd(records(1)).replace(b"DATA binary", b"DATA ascii"), "c.pcd")
