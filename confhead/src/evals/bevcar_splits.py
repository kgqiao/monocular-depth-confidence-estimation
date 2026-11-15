# confhead/src/evals/bevcar_splits.py
# Scene lists from BEVCar (Schramm et al., IROS 2024, arXiv 2403.11761)
# https://github.com/robot-learning-freiburg/BEVCar
# Rainy-night scenes are in NIGHT, not RAIN. Groups are disjoint.


DAY_SCENES = [
    "scene-XXXX",  # <- paste 111 names from custom_nuscenes_splits.py
    # ...
]

RAIN_SCENES = [
    "scene-XXXX",  # <- paste 24 names
    # ...
]

NIGHT_SCENES = [
    "scene-XXXX",  # <- paste 15 names
    # ...
]

# Lookup: scene name -> BEVCar condition bucket
BEVCAR_CONDITION: dict[str, str] = (
    {s: "day"   for s in DAY_SCENES}
  | {s: "rain"  for s in RAIN_SCENES}
  | {s: "night" for s in NIGHT_SCENES}
)

# assert len(BEVCAR_CONDITION) == 150, "scene lists overlap or are incomplete"


# Ran Custom Confidence head evaluations - 
# AUC 1.0