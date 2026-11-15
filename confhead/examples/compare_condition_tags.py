"""
Compare the cache's keyword-based condition tags against BEVCar's curated lists.
Run on CARC: python compare_condition_tags.py --cache /path/to/cache
"""

from confhead.cache.store import Cache

from evals.bevcar_splits import BEVCAR_CONDITION, RAIN_SCENES, NIGHT_SCENES
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
args = parser.parse_args()

cache = Cache(args.cache)

mismatches = []
for row in cache.rows(tag="val"):   # or "test" depending on your split naming
    scene = row["scene"]
    bev_cond = BEVCAR_CONDITION.get(scene)
    if bev_cond is None:
        continue  # not in BEVCar val set
    
    cache_cond  = row["condition"]   # "day" or "night" from condition_of()
    cache_weather = row["weather"]   # "clear" or "rain" from weather_of()
    desc = row["description"]
    
    # What would the cache call it ?
    cache_bucket = "night" if cache_cond == "night" else cache_weather  # "night", "rain", or "day"
    
    if cache_bucket != bev_cond:
        mismatches.append({
            "scene": scene,
            "description": desc,
            "cache_bucket": cache_bucket,
            "bevcar_bucket": bev_cond,
        })

print(f"\nTotal mismatches: {len(mismatches)}")
for m in mismatches:
    print(f"  {m['scene']}: cache={m['cache_bucket']}, bevcar={m['bevcar_bucket']}")
    print(f"    desc: {m['description'][:80]}")