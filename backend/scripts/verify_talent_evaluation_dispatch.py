from __future__ import annotations

import argparse
import json
import logging

from app.talent_decision_graph import DecisionContext
from app.talent_evaluation_runtime import graph
from utils.file_utils import dump_json

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in (
        "app.evidence_pack",
        "app.talent_tools",
        "app.talent_evaluation_dispatch",
        "app.talent_evaluation_runtime",
        __name__,
    ):
        logging.getLogger(logger_name).setLevel(level)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证第 15 课动态维度与 Subagent 分发链路")
    parser.add_argument("--tenant-id", default="course-demo")
    parser.add_argument("--region", default="上海")
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging(args.log_level)
    logger.info(
        "event=verification_started function=main tenant_id=%s region=%s "
        "max_concurrency=%s log_level=%s",
        args.tenant_id, args.region, args.max_concurrency, args.log_level,
    )

    result = graph.invoke(
        {
            "talent_request": {
                "original_text": f"筛选{args.region}且具备 RAG 落地与 Agent 评测经验的人才，协作交付经验优先",
                "source": "detailed_requirement",
                "hard_conditions": [
                    {"field": "region", "operator": "eq", "value": args.region}
                ],
                "semantic_conditions": [
                    {"requirement_id": "S1", "query": "RAG 项目落地经验", "required": True},
                    {"requirement_id": "S2", "query": "Agent 评测经验", "required": True},
                ],
                "evaluation_preferences": ["协作与交付经验优先"],
            },
            "query_plan": {
                "task_type": "find_talent",
                "filters": [
                    {"field": "region", "operator": "eq", "value": args.region}
                ],
            },
        },
        config={"max_concurrency": args.max_concurrency},
        context=DecisionContext(
            tenant_id=args.tenant_id,
            permission_scopes=("hr_private",),
        ),
    )

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_json(result, f"output/talent_evaluation_dispatch_result_{current_time}.json", ensure_ascii=False, indent=2)
    dimensions = [
        {
            "dimension_number": dimension_number,
            "weight_percent": item["weight_percent"],
            "retrieval_hints": item["retrieval_hints"],
        }
        for dimension_number, item in enumerate(result["dimensions"], start=1)
    ]
    status_counts: dict[str, int] = {}
    for item in result["branch_results"]:
        execution_status = item["execution_status"]
        status_counts[execution_status] = status_counts.get(execution_status, 0) + 1
    non_succeeded = [
        {
            "task_id": item["task_id"],
            "execution_status": item["execution_status"],
            "error_code": item["error_code"],
            "requirement_reasons": [
                requirement["reason"]
                for requirement in item["requirements"]
                if requirement["reason"] not in {"evidence_review", "no_accessible_hits"}
            ],
        }
        for item in result["branch_results"]
        if item["execution_status"] != "succeeded"
    ]

    print("[dimensions] " + json.dumps(dimensions, ensure_ascii=False))
    print(
        "[dispatch] "
        + json.dumps(
            {
                "candidates": result["candidate_ids"],
                "work_items": len(result["work_items"]),
                "max_concurrency": args.max_concurrency,
            },
            ensure_ascii=False,
        )
    )
    print(
        "[result] "
        + json.dumps(
            {
                "status": result["status"],
                "counts": status_counts,
                "non_succeeded": non_succeeded,
            },
            ensure_ascii=False,
        )
    )
    logger.info(
        "event=verification_completed function=main status=%s "
        "candidate_count=%s work_item_count=%s status_counts=%s",
        result["status"], len(result["candidate_ids"]),
        len(result["work_items"]), status_counts,
    )


if __name__ == "__main__":
    main()
