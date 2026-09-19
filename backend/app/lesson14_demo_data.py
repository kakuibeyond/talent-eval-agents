from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import JobDescription


def seed_job_descriptions(
    records: Sequence[dict[str, Any]],
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> dict[str, int]:
    inserted = 0
    updated = 0
    with session_factory() as db:
        for record in records:
            job = db.scalar(
                select(JobDescription).where(JobDescription.job_code == record["job_code"])
            )
            if job is None:
                db.add(JobDescription(**record))
                inserted += 1
                continue
            for field in ("tenant_id", "name", "content", "version", "status"):
                setattr(job, field, record[field])
            updated += 1
        db.commit()
    return {"inserted": inserted, "updated": updated}
