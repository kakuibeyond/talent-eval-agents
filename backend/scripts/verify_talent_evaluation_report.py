from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.talent_evaluation_runtime import report_graph as graph
from utils.file_utils import dump_json


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "lesson16"
    / "talent_evaluation_dispatch_result_20260921_205347.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证第 16 课证据评估与报告合成链路"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    return parser


def load_dispatch_result(input_path: Path) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    dimensions = payload.get("dimensions")
    branch_results = payload.get("branch_results")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("分发结果缺少 dimensions")
    if not isinstance(branch_results, list) or not branch_results:
        raise ValueError("分发结果缺少 branch_results")
    return {
        "dimensions": dimensions,
        "branch_results": branch_results,
    }


def run_verification(input_path: Path = DEFAULT_INPUT_PATH) -> dict:
    return graph.invoke(load_dispatch_result(input_path))


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = run_verification(args.input)

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_json(result, f"output/talent_evaluation_report_result_{current_time}.json", ensure_ascii=False, indent=2)

    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "confirmed_score": item["confirmed_score"],
            "possible_score": item["possible_score"],
            "evidence_coverage_percent": item["evidence_coverage_percent"],
            "confidence": item["confidence"],
        }
        for item in result["candidate_assessments"]
    ]
    status_counts: dict[str, int] = {}
    for candidate in result["candidate_assessments"]:
        for dimension in candidate["dimensions"]:
            status = dimension["evidence_status"]
            status_counts[status] = status_counts.get(status, 0) + 1

    print(
        "[ranking] "
        + json.dumps(
            {
                "order": result["ranking"],
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
    )
    print(
        "[evidence_status] "
        + json.dumps(status_counts, ensure_ascii=False, sort_keys=True)
    )
    print("[report]")
    print(result["report"])


if __name__ == "__main__":
    main()
