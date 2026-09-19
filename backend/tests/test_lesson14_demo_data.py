from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.lesson14_demo_data import seed_job_descriptions
from app.models import Base, JobDescription


def test_seed_job_descriptions_is_idempotent_and_updates_existing_rows():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = lambda: Session(engine)
    records = [
        {
            "job_code": "JD-AI-001",
            "tenant_id": "course-demo",
            "name": "高级 AI 应用工程师",
            "content": "负责企业知识库与 Agent 应用建设",
            "version": 1,
            "status": "active",
        },
        {
            "job_code": "JD-AI-002",
            "tenant_id": "course-demo",
            "name": "AI 应用工程师（平台方向）",
            "content": "负责模型平台与可观测性建设",
            "version": 1,
            "status": "active",
        },
    ]

    assert seed_job_descriptions(records, session_factory=session_factory) == {
        "inserted": 2,
        "updated": 0,
    }
    records[0]["version"] = 2
    assert seed_job_descriptions(records, session_factory=session_factory) == {
        "inserted": 0,
        "updated": 2,
    }

    with Session(engine) as db:
        jobs = db.scalars(select(JobDescription).order_by(JobDescription.job_code)).all()
        assert len(jobs) == 2
        assert jobs[0].version == 2
