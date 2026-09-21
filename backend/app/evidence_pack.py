"""Lesson 9: evidence organization. Model judgments remain reviewable hypotheses."""
from __future__ import annotations

import json
import logging
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class FactSource(StrictModel):
    chunk_id: str
    quote: str = Field(min_length=1, max_length=2000)


class ExtractedFact(StrictModel):
    event: str = Field(min_length=1, max_length=200)
    period: str = Field(pattern=r'^(?:unknown|\d{4}|\d{4}-\d{2}/\d{4}-\d{2})$')
    claim: str = Field(min_length=1, max_length=200)
    answer: Literal['yes', 'no']
    sources: list[FactSource] = Field(min_length=1, max_length=24)


class EvidenceExtraction(StrictModel):
    facts: list[ExtractedFact] = Field(default_factory=list, max_length=30)
    fully_supported: bool = False
    missing_information: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=10)


class EvidenceSourceValidationError(ValueError):
    """Machine-readable provenance failure without carrying source text."""

    def __init__(
        self,
        code: str,
        *,
        chunk_id: str,
        fact_index: int,
        source_index: int,
        quote_length: int,
        source_length: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.chunk_id = chunk_id
        self.fact_index = fact_index
        self.source_index = source_index
        self.quote_length = quote_length
        self.source_length = source_length


EXTRACTION_PROMPT = """根据给定查询要求整理材料中的事实陈述。
只返回符合 output_schema 的 JSON 对象，不输出说明文字或 Markdown 代码块。
只抽取与 requirement 相关的项目、时期、职责或成果，保留原文限定词，不补全未知事实。
event 填写材料明确出现的项目、岗位或事项名称。材料出现星河项目时填写星河项目，只有材料没有可识别事项时才填写 unknown。
period 对同一时间范围统一写成 YYYY-MM/YYYY-MM，例如 2025 年 1 月至 6 月写成 2025-01/2025-06。
claim 把查询要求改写成可回答是或否的命题，例如担任项目总负责人。
answer 仅使用 yes、no。明确满足命题为 yes，明确否定命题为 no，未回答命题的背景内容不生成事实。
同一事实的重复转述使用相同的 event、period、claim 和 answer。
输入材料使用 source_1、source_2 这类短 chunk_id，输出时必须完整复制，不得截断或改写。
每条事实必须有输入 chunk_id 和连续原文 quote，quote 不得改写。
fully_supported 仅表示当前材料直接覆盖当前要求全部要素，缺失、冲突时为 false。
没有缺失信息时 missing_information 必须返回空数组，不能返回空字符串。
原文引用正确不代表事实已经独立核实。不评分、不推荐候选人。"""


def model_extractor(model):
    def extract(requirement, sources):
        source_aliases = {
            f'source_{index}': row['chunk_id']
            for index, row in enumerate(sources, start=1)
        }
        payload = {
            'requirement': requirement['query'],
            'sources': [
                {'chunk_id': alias, 'content': row['content']}
                for alias, row in zip(source_aliases, sources, strict=True)
            ],
            'output_schema': EvidenceExtraction.model_json_schema(),
        }
        raw = model.with_structured_output(
            EvidenceExtraction,
            method='json_mode',
        ).invoke([
            ('system', EXTRACTION_PROMPT),
            ('user', json.dumps(payload, ensure_ascii=False)),
        ])
        data = raw if isinstance(raw, EvidenceExtraction) else EvidenceExtraction.model_validate(raw)
        facts = [
            fact.model_copy(
                update={
                    'sources': [
                        ref.model_copy(
                            update={
                                'chunk_id': source_aliases.get(ref.chunk_id, ref.chunk_id)
                            }
                        )
                        for ref in fact.sources
                    ]
                }
            )
            for fact in data.facts
        ]
        return data.model_copy(update={'facts': facts})
    return extract


def _validate_fact_sources(facts, sources):
    """Reject model facts that cannot point back to an input Chunk and exact quote."""
    by_id = {row['chunk_id']: row for row in sources}
    validated = []
    for fact_index, fact in enumerate(facts):
        refs = []
        for source_index, ref in enumerate(fact.sources):
            row = by_id.get(ref.chunk_id)
            quote = ref.quote.strip()
            if row is None:
                raise EvidenceSourceValidationError(
                    'unknown_chunk_id', chunk_id=ref.chunk_id,
                    fact_index=fact_index, source_index=source_index,
                    quote_length=len(quote),
                )
            if not quote:
                raise EvidenceSourceValidationError(
                    'blank_quote', chunk_id=ref.chunk_id,
                    fact_index=fact_index, source_index=source_index,
                    quote_length=0, source_length=len(row['content']),
                )
            quote_start = row['content'].find(quote)
            if quote_start < 0:
                raise EvidenceSourceValidationError(
                    'quote_not_found', chunk_id=ref.chunk_id,
                    fact_index=fact_index, source_index=source_index,
                    quote_length=len(quote), source_length=len(row['content']),
                )
            mapped = {
                'citation_id': row['citation_id'],
                'chunk_id': ref.chunk_id,
                'quote': quote,
                'quote_start': quote_start,
                'quote_end': quote_start + len(quote),
                'source_label': row.get('document_title') or row.get('source_label') or ref.chunk_id,
            }
            if mapped not in refs:
                refs.append(mapped)
        validated.append({**fact.model_dump(exclude={'sources'}), 'sources': refs})
    return validated


def _merge_duplicate_facts(facts):
    """Merge repeated descriptions while preserving every verified source."""
    merged, index_by_key = [], {}
    for fact in facts:
        key = (fact['event'], fact['period'], fact['claim'], fact['answer'])
        if key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append({**fact, 'sources': []})
        index = index_by_key[key]
        for ref in fact['sources']:
            if ref not in merged[index]['sources']:
                merged[index]['sources'].append(ref)
    return merged


def _find_conflicts(facts):
    """Find yes/no answers about the same event, period and claim."""
    conflicts = []
    for left_index, left in enumerate(facts):
        for right_index in range(left_index + 1, len(facts)):
            right = facts[right_index]
            same_scope = all(left[key] == right[key] and left[key] != 'unknown'
                for key in ('event', 'period', 'claim'))
            opposite_answers = {left['answer'], right['answer']} == {'yes', 'no'}
            if same_scope and opposite_answers:
                conflicts.append([left_index, right_index])
    return conflicts


def _evidence_status(*, facts, conflicts, fully_supported, missing_information):
    if conflicts:
        return 'conflicting'
    if (any(fact['answer'] == 'yes' for fact in facts) and fully_supported
            and not missing_information):
        return 'sufficient'
    if facts:
        return 'partial'
    return 'missing'


def build_evidence_packs(*, candidate_ids, requirements, sources, extract):
    logger.info(
        'event=evidence_pack_started function=build_evidence_packs '
        'candidate_count=%s requirement_count=%s source_count=%s',
        len(candidate_ids), len(requirements), len(sources),
    )
    packs = []
    for candidate_id in dict.fromkeys(candidate_ids):
        items = []
        for requirement in requirements:
            rows = list({row['chunk_id']: row for row in sources
                if row['candidate_id'] == candidate_id
                and requirement['requirement_id'] in row['requirement_ids']}.values())
            item = {**requirement, 'status': 'missing', 'reason': 'no_accessible_hits',
                    'extraction_status': 'not_run', 'facts': [], 'conflicts': [],
                    'missing_information': [], 'citations': rows}
            source_chunk_ids = [row['chunk_id'] for row in rows]
            logger.info(
                'event=evidence_requirement_started function=build_evidence_packs '
                'candidate_id=%s requirement_id=%s query_chars=%s '
                'source_count=%s source_chunk_ids=%s',
                candidate_id, requirement['requirement_id'],
                len(requirement.get('query', '')), len(rows), source_chunk_ids,
            )
            if rows:
                try:
                    raw = extract(requirement, rows)
                    data = raw if isinstance(raw, EvidenceExtraction) else EvidenceExtraction.model_validate(raw)
                    logger.info(
                        'event=evidence_model_extraction_completed '
                        'function=build_evidence_packs candidate_id=%s '
                        'requirement_id=%s fact_count=%s missing_count=%s '
                        'fully_supported=%s',
                        candidate_id, requirement['requirement_id'], len(data.facts),
                        len(data.missing_information), data.fully_supported,
                    )
                    facts = _merge_duplicate_facts(_validate_fact_sources(data.facts, rows))
                    conflicts = _find_conflicts(facts)
                    status = _evidence_status(facts=facts, conflicts=conflicts,
                        fully_supported=data.fully_supported, missing_information=data.missing_information)
                    item.update(status=status, reason='evidence_review' if facts else 'no_relevant_evidence',
                        extraction_status='succeeded', facts=facts, conflicts=conflicts,
                        missing_information=data.missing_information)
                except Exception as exc:
                    # Never return provider errors, raw prompts or document contents in errors.
                    if isinstance(exc, EvidenceSourceValidationError):
                        logger.error(
                            'event=evidence_extraction_failed '
                            'function=_validate_fact_sources stage=source_validation '
                            'error_type=%s error_code=%s candidate_id=%s '
                            'requirement_id=%s chunk_id=%s fact_index=%s '
                            'source_index=%s quote_length=%s source_length=%s '
                            'source_count=%s source_chunk_ids=%s',
                            type(exc).__name__, exc.code, candidate_id,
                            requirement['requirement_id'], exc.chunk_id,
                            exc.fact_index, exc.source_index, exc.quote_length,
                            exc.source_length, len(rows), source_chunk_ids,
                        )
                    elif isinstance(exc, ValidationError):
                        issue_types = [item['type'] for item in exc.errors()]
                        issue_locations = [
                            '.'.join(str(part) for part in item['loc'])
                            for item in exc.errors()
                        ]
                        logger.error(
                            'event=evidence_extraction_failed '
                            'function=EvidenceExtraction.model_validate '
                            'stage=structured_output_validation error_type=%s '
                            'error_code=schema_validation_failed candidate_id=%s '
                            'requirement_id=%s issue_types=%s issue_locations=%s '
                            'source_count=%s source_chunk_ids=%s',
                            type(exc).__name__, candidate_id,
                            requirement['requirement_id'], issue_types,
                            issue_locations, len(rows), source_chunk_ids,
                        )
                    else:
                        logger.error(
                            'event=evidence_extraction_failed function=extract '
                            'stage=model_extraction error_type=%s '
                            'error_code=model_extraction_failed candidate_id=%s '
                            'requirement_id=%s query_chars=%s source_count=%s '
                            'source_chunk_ids=%s',
                            type(exc).__name__, candidate_id,
                            requirement['requirement_id'],
                            len(requirement.get('query', '')), len(rows),
                            source_chunk_ids,
                            exc_info=logger.isEnabledFor(logging.DEBUG),
                        )
                    item.update(status='partial', reason='extraction_failed', extraction_status='failed')
            logger.info(
                'event=evidence_requirement_completed function=build_evidence_packs '
                'candidate_id=%s requirement_id=%s status=%s reason=%s '
                'extraction_status=%s fact_count=%s conflict_count=%s '
                'missing_count=%s',
                candidate_id, requirement['requirement_id'], item['status'],
                item['reason'], item['extraction_status'], len(item['facts']),
                len(item['conflicts']), len(item['missing_information']),
            )
            items.append(item)
        packs.append({'schema_version': '2.0', 'candidate_id': candidate_id, 'requirements': items})
    return packs
