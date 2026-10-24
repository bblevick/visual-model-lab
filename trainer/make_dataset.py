"""Original synthetic dispatch notices; split by event text before training."""

import json
from pathlib import Path

streets = [
    "Lantern Lane",
    "Orchard Row",
    "Market Bridge",
    "South Bridge",
    "North Bridge",
    "Moss Street",
    "Copper Alley",
    "Juniper Walk",
    "Canal Street",
    "Willow Way",
    "Amber Road",
    "Fern Passage",
]
conditions = [
    ("closed", "find another route"),
    ("reopened", "resume the original route"),
    ("flooded", "wait at the nearest safe intersection"),
    ("clear", "continue the delivery"),
]
rows = []
for i, street in enumerate(streets):
    for j, (condition, action) in enumerate(conditions):
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write a short dispatch notice: {street} is {condition}.",
                    },
                    {
                        "role": "assistant",
                        "content": f"NOTICE: {street} is {condition}. NEXT: {action}.",
                    },
                ]
            }
        )
# Streets in held-out splits never appear in the training split.
root = Path(__file__).parent / "data"
for name, subset in [("train", rows[:32]), ("valid", rows[32:40]), ("test", rows[40:])]:
    (root / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in subset))
