# confhead/src/evals/bevcar_splits.py
# Scene lists from BEVCar (Schramm et al., IROS 2024, arXiv 2403.11761)
# https://github.com/robot-learning-freiburg/BEVCar
# Source: custom_nuscenes_splits.py in the BEVCar repo
# val_day: 111 scenes (day, no rain)
# val_rain:  24 scenes (rainy, day or night)
# val_night: 15 scenes (night, no rain)
# Groups are disjoint; total = 150 val scenes.

DAY_SCENES = [
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
    'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036',
    'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094',
    'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
    'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104',
    'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109',
    'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270',
    'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
    'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330',
    'scene-0331', 'scene-0332', 'scene-0344', 'scene-0345', 'scene-0346',
    'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523',
    'scene-0524', 'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555',
    'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560',
    'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565',
    'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778',
    'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784',
    'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798',
    'scene-0799', 'scene-0800', 'scene-0802', 'scene-0916', 'scene-0917',
    'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923',
    'scene-0924', 'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928',
    'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962', 'scene-0963',
    'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971',
    'scene-0972',
]  # 111 scenes

RAIN_SCENES = [
    'scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630',
    'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
    'scene-0637', 'scene-0638', 'scene-0904', 'scene-0905', 'scene-0906',
    'scene-0907', 'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911',
    'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
]  # 24 scenes

NIGHT_SCENES = [
    'scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063',
    'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067', 'scene-1068',
    'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073',
]  # 15 scenes

assert len(DAY_SCENES) == 111,  f"Expected 111 day scenes, got {len(DAY_SCENES)}"
assert len(RAIN_SCENES) == 24,  f"Expected 24 rain scenes, got {len(RAIN_SCENES)}"
assert len(NIGHT_SCENES) == 15, f"Expected 15 night scenes, got {len(NIGHT_SCENES)}"
assert len(set(DAY_SCENES) | set(RAIN_SCENES) | set(NIGHT_SCENES)) == 150, \
    "Scene lists overlap or are incomplete"

# Lookup: scene name -> BEVCar condition bucket
BEVCAR_CONDITION: dict[str, str] = (
    {s: "day"   for s in DAY_SCENES}
  | {s: "rain"  for s in RAIN_SCENES}
  | {s: "night" for s in NIGHT_SCENES}
)