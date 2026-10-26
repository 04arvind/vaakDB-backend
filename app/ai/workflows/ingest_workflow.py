"""Raw text -> extracted, validated records -> insert into a registered target.

The LLM only fills in field values. Which table, which columns and who owns the
rows are decided by the write target registered in db/writer.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.ai.agents.extractor_agent import ExtractorAgent
from app.ai.db.writer import get_target, insert_records
from app.ai.schemas.llm_io import Usage

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    target: str
    dry_run: bool
    records: list[dict[str, Any]] = field(default_factory=list)
    inserted_ids: list[int | str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _is_empty(record: BaseModel) -> bool:
    return all(v in (None, "", [], {}) for v in record.model_dump().values())


async def run_ingest(
    text: str,
    target: str,
    *,
    user_id: str,
    dry_run: bool = False,
    provider: str | None = None,
) -> IngestResult:
    """Raises UnknownTargetError, AgentError (extraction failed) or ValueError (bad data)."""
    write_target = get_target(target)  # fail fast, before paying for an LLM call

    agent = ExtractorAgent(provider=provider)
    items = await agent.extract_many(text, write_target.model, hints=write_target.hints)
    items = [i for i in items if not _is_empty(i)]
    if len(items) > write_target.max_records:
        raise ValueError(
            f"Found {len(items)} records; the limit is {write_target.max_records} per request."
        )

    records = [
        {k: v for k, v in i.model_dump(mode="json").items() if k != write_target.owner_column}
        for i in items
    ]
    result = IngestResult(target=target, dry_run=dry_run, records=records, usage=agent.usage)
    if dry_run or not items:
        return result

    result.inserted_ids = await insert_records(target, items, user_id=user_id)
    log.info("Ingest into %s: %d extracted, %d inserted",
             target, len(items), len(result.inserted_ids))
    return result
