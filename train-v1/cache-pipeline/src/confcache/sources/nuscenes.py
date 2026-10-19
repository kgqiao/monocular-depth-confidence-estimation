"""nuScenes as a cache source, on top of the confhead loader.

Neighbor images are the 12 Hz camera sweeps around each keyframe image,
reached through the prev and next links of sample_data. Lidar and radar
sweeps are the keyframe sweep and the sweeps before it, at most
`points.lidar_sweeps` and `points.radar_sweeps` of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from confhead.nuscenes import NuScenes, pose_to_mat

from ..config import PointsConfig, SourceConfig, TemporalConfig
from .base import Item, PointFile, View


def weather_of(description: str) -> str:
    return "rain" if "rain" in description.lower() else "clear"


class NuScenesSource:
    name = "nuscenes"

    def __init__(self, source: SourceConfig, points: PointsConfig, temporal: TemporalConfig):
        if not source.dataroot:
            raise SystemExit("Set source.dataroot or $NUSCENES_DATAROOT: the folder that holds "
                             f"{source.version}/, samples/ and sweeps/.")
        dataroot = Path(source.dataroot)
        if not dataroot.is_dir():
            raise SystemExit(f"nuScenes dataroot {dataroot} does not exist.")
        self.cfg = source
        self.points = points
        self.offsets = sorted(temporal.offsets) if temporal.enabled else []
        self.nusc = NuScenes(dataroot, source.version)
        self.skipped_scenes = [s["name"] for s in self.nusc.skipped_scenes]
        self._scenes = [s for s in self.nusc.scenes if not source.scenes or s["name"] in source.scenes]
        missing = set(source.scenes) - {s["name"] for s in self.nusc.table["scene"]}
        if missing:
            raise SystemExit(f"source.scenes: {sorted(missing)} are not in {source.version}.")

    def scene_names(self) -> list[str]:
        return [s["name"] for s in self._scenes]

    def items(self) -> list[Item]:
        out = []
        for scene in self._scenes:
            samples = self.nusc.samples_of_scene[scene["token"]]
            if self.cfg.max_samples_per_scene:
                samples = samples[: self.cfg.max_samples_per_scene]
            for index, sample in enumerate(samples):
                frame = self.nusc.frame(sample["token"])
                for channel in self.cfg.cameras:
                    if channel in frame.camera_sds:
                        out.append(self._item(scene, index, frame, channel))
        return out

    # ----- one item ------------------------------------------------------

    def _item(self, scene: dict, index: int, frame, channel: str) -> Item:
        nusc = self.nusc
        cam_sd = frame.camera_sds[channel]
        cam = nusc.camera(frame, channel)
        view = self._view(cam_sd)
        ref_to_cam = cam.ref_to_cam
        t_cam = cam_sd["timestamp"]

        def point_files(first_sd: dict, count: int, fmt: str) -> tuple[PointFile, ...]:
            files, sd = [], first_sd
            while sd is not None and len(files) < count:
                path = nusc.dataroot / sd["filename"]
                if path.exists():
                    files.append(PointFile(
                        path=path, format=fmt,
                        sensor_to_cam=ref_to_cam @ nusc.sensor_to_ref(frame, sd),
                        dt=(sd["timestamp"] - t_cam) / 1e6,
                        sweep=len(files),
                    ))
                sd = nusc.get("sample_data", sd["prev"]) if sd["prev"] else None
            return tuple(files)

        lidar = point_files(frame.lidar_sd, self.points.lidar_sweeps, "nuscenes-lidar")
        radar = tuple(
            pf for sd in frame.radar_sds.values()
            for pf in point_files(sd, self.points.radar_sweeps, "nuscenes-radar")
        )
        return Item(
            key=f"{scene['name']}_{index:03d}_{channel}",
            scene=scene["name"],
            condition=nusc.condition(scene),
            weather=weather_of(scene["description"]),
            channel=channel,
            sample_token=frame.sample["token"],
            index=index,
            view=view,
            neighbors=self._neighbors(cam_sd),
            lidar=lidar,
            radar=radar,
        )

    def _view(self, sd: dict) -> View:
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = pose_to_mat(self.nusc.get("ego_pose", sd["ego_pose_token"]))
        return View(
            token=sd["token"],
            image_path=self.nusc.dataroot / sd["filename"],
            width=sd["width"],
            height=sd["height"],
            K=np.array(cs["camera_intrinsic"]),
            cam_to_global=ego @ pose_to_mat(cs),
            timestamp=sd["timestamp"],
        )

    def _neighbors(self, cam_sd: dict) -> tuple[tuple[int, View], ...]:
        out = []
        for direction, link in ((-1, "prev"), (1, "next")):
            wanted = sorted((o for o in self.offsets if o * direction > 0), key=abs)
            sd, step = cam_sd, 0
            for offset in wanted:
                while sd is not None and step < abs(offset):
                    sd = self.nusc.get("sample_data", sd[link]) if sd[link] else None
                    step += 1
                if sd is None:
                    break
                if (self.nusc.dataroot / sd["filename"]).exists():
                    out.append((offset, self._view(sd)))
        return tuple(sorted(out, key=lambda p: p[0]))

