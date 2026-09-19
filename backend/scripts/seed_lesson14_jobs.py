from __future__ import annotations

import json
from pathlib import Path

from app.lesson14_demo_data import seed_job_descriptions


SAMPLE_PATH = Path(__file__).resolve().parents[1] / "samples" / "lesson14" / "job_descriptions.json"


def main() -> None:
    records = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    result = seed_job_descriptions(records)
    print(json.dumps({**result, "total": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
