import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAA.Geometry.config import SCHWARTZ_CIRCUMPLEX_ORDER


SOURCE_PATH = Path("CAA/value_data/schwartz_relations.json")
TARGET_PATH = Path("CAA/value_data/schwartz_relations-new.json")


def circular_step_distance(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


def cosine_circumplex_matrix() -> dict:
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    step_angle_deg = 360.0 / n
    matrix = {}

    for i, value_i in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        row = {}
        for j, value_j in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            step = circular_step_distance(i, j, n)
            similarity = math.cos(math.radians(step * step_angle_deg))
            if abs(similarity) < 1e-12:
                similarity = 0.0
            row[value_j] = round(similarity, 6)
        matrix[value_i] = row

    return matrix


def main() -> None:
    with SOURCE_PATH.open() as f:
        data = json.load(f)

    data["basic_value_relationship_matrix"] = cosine_circumplex_matrix()

    with TARGET_PATH.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Wrote {TARGET_PATH}")


if __name__ == "__main__":
    main()
