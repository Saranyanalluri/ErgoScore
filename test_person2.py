import json

from app.person1_adapter import convert_keypoints
from app.scoring_pipeline import process_keypoints


INPUT = "data/keypoints/samplebabbu_keypoints.json"


with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Input frames:", len(data["frames"]))

converted = convert_keypoints(data)

print("Converted frames:", len(converted["frames"]))

results = process_keypoints(
    converted["frames"],
    sitting=False
)

print("Scored frames:", len(results))

if results:
    print("\nFirst result:")
    print(results[0])

print("\nPERSON 2 SUCCESS")