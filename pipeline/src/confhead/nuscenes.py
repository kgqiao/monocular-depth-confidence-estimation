"""Minimal nuScenes reader, ported from autonomous-vehicle-sandbox.

Loads the JSON tables of one split, indexes them by token, and puts every
sensor of a keyframe into one reference frame: the ego frame at the
LIDAR_TOP timestamp of that keyframe.

Frames follow nuScenes conventions: x forward, y left, z up. Quaternions are
[w, x, y, z]. Every 4x4 matrix maps a column vector from the frame named first
to the frame named second, e.g. `sensor_to_ref`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATAROOT_ENV = "NUSCENES_DATAROOT"

TABLES = ["scene", "log", "sensor", "calibrated_sensor", "sample", "sample_data", "ego_pose"]

CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
RADAR_CHANNELS = [
    "RADAR_FRONT_LEFT", "RADAR_FRONT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]
LIDAR_CHANNEL = "LIDAR_TOP"

RADAR_PCD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dyn_prop", "i1"), ("id", "<i2"),
    ("rcs", "<f4"), ("vx", "<f4"), ("vy", "<f4"), ("vx_comp", "<f4"), ("vy_comp", "<f4"),
    ("is_quality_valid", "i1"), ("ambig_state", "i1"), ("x_rms", "i1"), ("y_rms", "i1"),
    ("invalid_state", "i1"), ("pdh0", "i1"), ("vx_rms", "i1"), ("vy_rms", "i1"),
])


def resolve_dataroot(arg: str | None) -> Path:
    """The --dataroot flag, else $NUSCENES_DATAROOT."""
    value = arg or os.environ.get(DATAROOT_ENV)
    if not value:
        raise SystemExit(f"Set {DATAROOT_ENV} or pass --dataroot: the nuScenes folder that holds v1.0-mini/.")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"nuScenes dataroot {path} does not exist.")
    return path


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_mat(record) -> np.ndarray:
    """calibrated_sensor or ego_pose record -> 4x4 homogeneous."""
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(record["rotation"])
    m[:3, 3] = record["translation"]
    return m


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an (N,3) array of points."""
    return pts @ m[:3, :3].T + m[:3, 3]


def parse_radar_pcd(raw: bytes, name: str) -> np.ndarray:
    """Binary PCD bytes -> structured array of RADAR_PCD_DTYPE records.

    The header must declare exactly the fields, sizes and types of
    RADAR_PCD_DTYPE; anything else fails with the file name.
    """
    marker = b"DATA binary\n"
    if marker not in raw:
        raise ValueError(f"{name}: not a binary PCD file")
    header_end = raw.index(marker) + len(marker)
    header = {}
    for line in raw[:header_end].decode("ascii", errors="replace").splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition(" ")
            header[key] = value
    names = list(RADAR_PCD_DTYPE.names)
    expected = {
        "FIELDS": names,
        "SIZE": [str(RADAR_PCD_DTYPE[n].itemsize) for n in names],
        "TYPE": ["F" if RADAR_PCD_DTYPE[n].kind == "f" else "I" for n in names],
        "COUNT": ["1"] * len(names),
    }
    for key, want in expected.items():
        got = header.get(key, "").split()
        if got != want:
            raise ValueError(f"{name}: PCD {key} is {got}, expected {want}")
    points = header.get("POINTS", "").strip()
    if not points.isdigit():
        raise ValueError(f"{name}: PCD POINTS is {points!r}, expected a count")
    n = int(points)
    if len(raw) - header_end < n * RADAR_PCD_DTYPE.itemsize:
        raise ValueError(f"{name}: shorter than its {n} points; the file is truncated")
    return np.frombuffer(raw, dtype=RADAR_PCD_DTYPE, count=n, offset=header_end)


def condition_of(description: str) -> str:
    return "night" if "night" in description.lower() else "day"


@dataclass
class Frame:
    """One keyframe. `ref_to_global` is the ego pose at the lidar timestamp."""

    sample: dict
    ref_to_global: np.ndarray
    lidar_sd: dict
    camera_sds: dict[str, dict]
    radar_sds: dict[str, dict]


@dataclass
class Camera:
    channel: str
    sd_token: str
    image_path: Path
    width: int
    height: int
    K: np.ndarray  # (3,3)
    ref_to_cam: np.ndarray  # (4,4)


class NuScenes:
    def __init__(self, dataroot: Path, version: str = "v1.0-mini"):
        self.dataroot = Path(dataroot)
        self.version = version
        table_dir = self.dataroot / version
        if not table_dir.is_dir():
            raise SystemExit(f"{table_dir} does not exist; the dataroot must hold the {version}/ tables.")
        self.table: dict[str, list[dict]] = {
            t: json.loads((table_dir / f"{t}.json").read_text()) for t in TABLES
        }
        self.by_token: dict[str, dict[str, dict]] = {
            t: {r["token"]: r for r in rows} for t, rows in self.table.items()
        }
        self._index()
        self._keep_scenes_on_disk()

    def get(self, table: str, token: str) -> dict:
        return self.by_token[table][token]

    def _index(self) -> None:
        self.channel_of_cs: dict[str, str] = {}
        for cs in self.table["calibrated_sensor"]:
            self.channel_of_cs[cs["token"]] = self.get("sensor", cs["sensor_token"])["channel"]

        self.keyframe_sd: dict[str, dict[str, dict]] = {}
        for sd in self.table["sample_data"]:
            if sd["is_key_frame"]:
                channel = self.channel_of_cs[sd["calibrated_sensor_token"]]
                self.keyframe_sd.setdefault(sd["sample_token"], {})[channel] = sd

        self.samples_of_scene: dict[str, list[dict]] = {}
        for scene in self.table["scene"]:
            samples = []
            token = scene["first_sample_token"]
            while token:
                sample = self.get("sample", token)
                samples.append(sample)
                token = sample["next"]
            self.samples_of_scene[scene["token"]] = samples

    def _keep_scenes_on_disk(self) -> None:
        """The tables list every scene of the split; a partial download may
        carry the files of only some. Keep the scenes whose keyframe files all
        exist; list the others in `skipped_scenes`."""
        def present(scene: dict) -> bool:
            return all(
                (self.dataroot / sd["filename"]).exists()
                for sample in self.samples_of_scene[scene["token"]]
                for sd in self.keyframe_sd[sample["token"]].values()
            )

        self.scenes = [s for s in self.table["scene"] if present(s)]
        kept = {s["token"] for s in self.scenes}
        self.skipped_scenes = [s for s in self.table["scene"] if s["token"] not in kept]

    def condition(self, scene: dict) -> str:
        return condition_of(scene["description"])

    # ----- frames -------------------------------------------------------

    def frame(self, sample_token: str) -> Frame:
        sample = self.get("sample", sample_token)
        sds = self.keyframe_sd[sample_token]
        lidar_sd = sds[LIDAR_CHANNEL]
        return Frame(
            sample=sample,
            ref_to_global=pose_to_mat(self.get("ego_pose", lidar_sd["ego_pose_token"])),
            lidar_sd=lidar_sd,
            camera_sds={c: sds[c] for c in CAMERA_CHANNELS if c in sds},
            radar_sds={c: sds[c] for c in RADAR_CHANNELS if c in sds},
        )

    def sensor_to_ref(self, frame: Frame, sd: dict) -> np.ndarray:
        """Sensor frame at the sensor's own timestamp -> reference ego frame."""
        cs = self.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego_to_global = pose_to_mat(self.get("ego_pose", sd["ego_pose_token"]))
        return np.linalg.inv(frame.ref_to_global) @ ego_to_global @ pose_to_mat(cs)

    def camera(self, frame: Frame, channel: str) -> Camera:
        sd = frame.camera_sds[channel]
        cs = self.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return Camera(
            channel=channel,
            sd_token=sd["token"],
            image_path=self.dataroot / sd["filename"],
            width=sd["width"],
            height=sd["height"],
            K=np.array(cs["camera_intrinsic"]),
            ref_to_cam=np.linalg.inv(self.sensor_to_ref(frame, sd)),
        )

    # ----- point clouds -------------------------------------------------

    def read_lidar(self, frame: Frame) -> np.ndarray:
        """(N,5) float32: x y z intensity ring, in the reference ego frame."""
        path = self.dataroot / frame.lidar_sd["filename"]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        out = pts.copy()
        out[:, :3] = apply(self.sensor_to_ref(frame, frame.lidar_sd), pts[:, :3])
        return out

    def read_radar(self, frame: Frame, sd: dict) -> np.ndarray:
        """(N,3) float64 xyz in the reference ego frame.

        Filtering follows the devkit defaults: invalid_state == 0,
        ambig_state == 3, and dyn_prop 0 to 6 (7, "stopped", is dropped).
        """
        path = self.dataroot / sd["filename"]
        rec = parse_radar_pcd(path.read_bytes(), path.name)
        rec = rec[(rec["invalid_state"] == 0) & (rec["ambig_state"] == 3) & (rec["dyn_prop"] < 7)]
        xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        return apply(self.sensor_to_ref(frame, sd), xyz)

    def read_all_radar(self, frame: Frame) -> np.ndarray:
        parts = [self.read_radar(frame, sd) for sd in frame.radar_sds.values()]
        return np.concatenate(parts) if parts else np.empty((0, 3))
