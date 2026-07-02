import json

with open(
    "data/processed/chunks.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

print("Parents:", len(data["parents"]))
print("Children:", len(data["children"]))

parent_ids = set()

for p in data["parents"]:
    parent_ids.add(p["chunk_id"])

bad = 0

for child in data["children"]:
    if child["parent_id"] not in parent_ids:
        bad += 1

print("Broken parent references:", bad)