"""nuScenes, read with numpy alone (no nuscenes-devkit).

`NuScenes` loads the JSON tables of one version, keeps the scenes whose
keyframe files are on disk, and turns them into `Item`s: one per camera image
of a keyframe. Neighbor images are the 12 Hz camera sweeps around each
keyframe image, reached through the prev and next links of sample_data. Radar
sweeps are the keyframe sweep and the sweeps before it.

Ego frame: x forward, y left, z up.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import pose_to_mat
from .items import Item, PointFile, View

DATAROOT_ENV = "NUSCENES_DATAROOT"

TABLES = ["scene", "log", "sensor", "calibrated_sensor", "sample", "sample_data", "ego_pose"]

CAMERA_CHANNELS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
RADAR_CHANNELS = (
    "RADAR_FRONT_LEFT", "RADAR_FRONT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
)
LIDAR_CHANNEL = "LIDAR_TOP"

RADAR_PCD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dyn_prop", "i1"), ("id", "<i2"),
    ("rcs", "<f4"), ("vx", "<f4"), ("vy", "<f4"), ("vx_comp", "<f4"), ("vy_comp", "<f4"),
    ("is_quality_valid", "i1"), ("ambig_state", "i1"), ("x_rms", "i1"), ("y_rms", "i1"),
    ("invalid_state", "i1"), ("pdh0", "i1"), ("vx_rms", "i1"), ("vy_rms", "i1"),
])


# ----- point files ----------------------------------------------------------


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


def read_lidar(path: Path) -> np.ndarray:
    """A LIDAR_TOP .pcd.bin -> (N, 3) float64 xyz in the sensor frame."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)


def read_radar(path: Path) -> np.ndarray:
    """A radar .pcd -> (N, 3) float64 xyz in the sensor frame, after the devkit's
    default filter: invalid_state 0, ambig_state 3, dyn_prop 0 to 6 (7, "stopped", is dropped)."""
    rec = parse_radar_pcd(Path(path).read_bytes(), Path(path).name)
    rec = rec[(rec["invalid_state"] == 0) & (rec["ambig_state"] == 3) & (rec["dyn_prop"] < 7)]
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)


# ----- scenes ---------------------------------------------------------------


def condition_of(description: str) -> str:
    return "night" if "night" in description.lower() else "day"


def weather_of(description: str) -> str:
    return "rain" if "rain" in description.lower() else "clear"


@dataclass(frozen=True)
class Scene:
    token: str
    name: str
    description: str
    condition: str  # day or night
    weather: str  # clear or rain
    samples: tuple[str, ...]  # keyframe sample tokens in time order


class NuScenes:
    """`NuScenes(dataroot).items(cameras=["CAM_FRONT"])`.

    dataroot: the folder that holds `<version>/`, `samples/` and `sweeps/`.
    """

    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini"):
        self.dataroot = Path(dataroot).expanduser()
        self.version = version
        table_dir = self.dataroot / version
        if not table_dir.is_dir():
            raise FileNotFoundError(f"{table_dir} does not exist; the dataroot must hold the {version}/ tables.")
        self.table: dict[str, list[dict]] = {
            t: json.loads((table_dir / f"{t}.json").read_text()) for t in TABLES
        }
        self._by_token = {t: {r["token"]: r for r in rows} for t, rows in self.table.items()}
        self._channel_of_cs = {
            cs["token"]: self.get("sensor", cs["sensor_token"])["channel"] for cs in self.table["calibrated_sensor"]
        }
        self._keyframe_sd: dict[str, dict[str, dict]] = {}
        for sd in self.table["sample_data"]:
            if sd["is_key_frame"]:
                self._keyframe_sd.setdefault(sd["sample_token"], {})[self._channel_of_cs[sd["calibrated_sensor_token"]]] = sd

        every = [self._scene(s) for s in self.table["scene"]]
        # The tables list every scene of the version; a partial download may carry the files of only some.
        self.scenes: list[Scene] = [s for s in every if self._on_disk(s)]
        self.skipped_scenes: list[Scene] = [s for s in every if s not in self.scenes]

    def get(self, table: str, token: str) -> dict:
        return self._by_token[table][token]

    def scene(self, name: str) -> Scene:
        for s in self.scenes:
            if s.name == name:
                return s
        raise KeyError(f"no scene {name!r} on disk in {self.version}")

    def _scene(self, record: dict) -> Scene:
        samples, token = [], record["first_sample_token"]
        while token:
            samples.append(token)
            token = self.get("sample", token)["next"]
        return Scene(token=record["token"], name=record["name"], description=record["description"],
                     condition=condition_of(record["description"]), weather=weather_of(record["description"]),
                     samples=tuple(samples))

    def _on_disk(self, scene: Scene) -> bool:
        return all((self.dataroot / sd["filename"]).exists()
                   for s in scene.samples for sd in self._keyframe_sd[s].values())

    # ----- items ---------------------------------------------------------

    def items(self, scenes: Iterable[str] | None = None, cameras: Iterable[str] = CAMERA_CHANNELS,
              offsets: Iterable[int] = (), radar_sweeps: int = 1, max_samples_per_scene: int = 0) -> list[Item]:
        """Every camera image of every keyframe, in scene order, then keyframe order, then camera order.

        scenes: scene names; None for every scene on disk.
        offsets: neighbor camera sweeps to attach, e.g. (-2, -1, 1, 2); an offset past the scene end is left out.
        radar_sweeps: sweeps per radar, the keyframe sweep and the ones before it.
        max_samples_per_scene: 0 for every keyframe.
        """
        cameras = list(cameras)
        unknown = set(cameras) - set(CAMERA_CHANNELS)
        if unknown:
            raise ValueError(f"unknown camera channels {sorted(unknown)}")
        chosen = self.scenes
        if scenes is not None:
            names = set(scenes)
            missing = names - {s["name"] for s in self.table["scene"]}
            if missing:
                raise ValueError(f"scenes {sorted(missing)} are not in {self.version}")
            chosen = [s for s in self.scenes if s.name in names]
        out = []
        for scene in chosen:
            samples = scene.samples[:max_samples_per_scene] if max_samples_per_scene else scene.samples
            for index in range(len(samples)):
                for channel in cameras:
                    if channel in self._keyframe_sd[samples[index]]:
                        out.append(self.item(scene, index, channel, offsets, radar_sweeps))
        return out

    def item(self, scene: Scene, index: int, channel: str, offsets: Iterable[int] = (),
             radar_sweeps: int = 1) -> Item:
        sds = self._keyframe_sd[scene.samples[index]]
        cam_sd = sds[channel]
        view = self._view(cam_sd)
        global_to_cam = np.linalg.inv(view.cam_to_global)

        def point_files(first: dict, count: int, read) -> tuple[PointFile, ...]:
            files, sd = [], first
            while sd is not None and len(files) < count:
                path = self.dataroot / sd["filename"]
                if path.exists():
                    files.append(PointFile(
                        path=path, read=read,
                        sensor_to_cam=global_to_cam @ self._sensor_to_global(sd),
                        dt=(sd["timestamp"] - cam_sd["timestamp"]) / 1e6,
                        sweep=len(files),
                    ))
                sd = self.get("sample_data", sd["prev"]) if sd["prev"] else None
            return tuple(files)

        return Item(
            key=f"{scene.name}_{index:03d}_{channel}",
            scene=scene.name,
            condition=scene.condition,
            weather=scene.weather,
            channel=channel,
            sample_token=scene.samples[index],
            index=index,
            view=view,
            neighbors=self._neighbors(cam_sd, offsets),
            lidar=point_files(sds[LIDAR_CHANNEL], 1, read_lidar),
            radar=tuple(pf for c in RADAR_CHANNELS if c in sds
                        for pf in point_files(sds[c], radar_sweeps, read_radar)),
        )

    def _sensor_to_global(self, sd: dict) -> np.ndarray:
        """Sensor frame at the sensor's own timestamp -> global frame."""
        cs = self.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return pose_to_mat(self.get("ego_pose", sd["ego_pose_token"])) @ pose_to_mat(cs)

    def _view(self, sd: dict) -> View:
        cs = self.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return View(
            token=sd["token"],
            image_path=self.dataroot / sd["filename"],
            width=sd["width"],
            height=sd["height"],
            K=np.array(cs["camera_intrinsic"]),
            cam_to_global=self._sensor_to_global(sd),
            timestamp=sd["timestamp"],
        )

    def _neighbors(self, cam_sd: dict, offsets: Iterable[int]) -> tuple[tuple[int, View], ...]:
        out = []
        for direction, link in ((-1, "prev"), (1, "next")):
            wanted = sorted((o for o in set(offsets) if o * direction > 0), key=abs)
            sd, step = cam_sd, 0
            for offset in wanted:
                while sd is not None and step < abs(offset):
                    sd = self.get("sample_data", sd[link]) if sd[link] else None
                    step += 1
                if sd is None:
                    break
                if (self.dataroot / sd["filename"]).exists():
                    out.append((offset, self._view(sd)))
        return tuple(sorted(out, key=lambda p: p[0]))
