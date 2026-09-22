from __future__ import annotations


def _dimension(dimension_name: str, weight_percent: int):
    from app.talent_evaluation_dispatch import EvaluationDimension, ScoreAnchor

    return EvaluationDimension(
        name=dimension_name,
        definition=f"{dimension_name} 的可取证定义",
        weight_percent=weight_percent,
        evidence_requirements=["项目职责与交付结果"],
        score_anchors=[
            ScoreAnchor(score=0, description="有明确事实表明未达到要求"),
            ScoreAnchor(score=3, description="存在可验证的参与事实"),
            ScoreAnchor(score=5, description="存在完整且可验证的交付事实"),
        ],
        retrieval_hints=[f"{dimension_name} 项目经验"],
        source_requirement_ids=["S1"],
    )


def _branch_requirement(
    chunk_id: str | None = None,
    *,
    missing_information: list[str] | None = None,
):
    facts = []
    if chunk_id is not None:
        facts = [
            {
                "event": "项目",
                "period": "2025",
                "claim": "具备相关经历",
                "answer": "yes",
                "sources": [
                    {
                        "citation_id": chunk_id,
                        "chunk_id": chunk_id,
                        "quote": "主导项目上线",
                        "quote_start": 0,
                        "quote_end": 6,
                        "source_label": "项目复盘",
                    }
                ],
            }
        ]
    return {
        "requirement_id": "requirement:1",
        "query": "项目职责与交付结果",
        "status": "partial" if missing_information else ("sufficient" if facts else "missing"),
        "reason": "evidence_review" if facts else "no_accessible_hits",
        "extraction_status": "succeeded" if facts else "not_run",
        "facts": facts,
        "conflicts": [],
        "missing_information": missing_information or [],
        "citations": [],
    }


def test_aggregate_candidate_keeps_missing_dimension_unscored():
    from app.talent_evaluation_report import (
        DimensionAssessment,
        aggregate_candidate_assessments,
    )

    dimensions = [
        _dimension("rag_delivery", 60),
        _dimension("agent_evaluation", 40),
    ]
    assessments = [
        DimensionAssessment(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            evidence_status="sufficient",
            score=4,
            confidence="high",
            evidence_refs=["E1"],
            missing_items=[],
            conflicts=[],
            rationale="有可验证的 RAG 交付事实",
        ),
        DimensionAssessment(
            task_id="C001:2",
            candidate_id="C001",
            dimension_number=2,
            evidence_status="missing",
            score=None,
            confidence="unavailable",
            evidence_refs=[],
            missing_items=["缺少 Agent 评测材料"],
            conflicts=[],
            rationale="当前材料无法评分",
        ),
    ]

    result = aggregate_candidate_assessments(dimensions, assessments)

    assert result[0].confirmed_score == 48.0
    assert result[0].possible_score == 88.0
    assert result[0].evidence_coverage_percent == 60
    assert result[0].confidence == "low"
    assert result[0].dimensions[1].score is None


def test_validate_dimension_judgement_rejects_unavailable_evidence_ref():
    import pytest

    from app.talent_evaluation_dispatch import BranchEvidenceDraft
    from app.talent_evaluation_report import (
        DimensionJudgement,
        validate_dimension_judgement,
    )

    branch = BranchEvidenceDraft(
        task_id="C001:1",
        candidate_id="C001",
        dimension_number=1,
        execution_status="succeeded",
        requirements=[_branch_requirement("E1")],
    )
    judgement = DimensionJudgement(
        evidence_status="sufficient",
        score=5,
        confidence="high",
        evidence_refs=["E2"],
        missing_items=[],
        conflicts=[],
        rationale="完整主导 RAG 项目交付",
    )

    with pytest.raises(ValueError, match="E2"):
        validate_dimension_judgement(branch, judgement)


def test_validate_dimension_judgement_rejects_score_for_missing_evidence():
    import pytest

    from app.talent_evaluation_dispatch import BranchEvidenceDraft
    from app.talent_evaluation_report import (
        DimensionJudgement,
        validate_dimension_judgement,
    )

    branch = BranchEvidenceDraft(
        task_id="C001:2",
        candidate_id="C001",
        dimension_number=2,
        execution_status="succeeded",
        requirements=[
            _branch_requirement(missing_information=["缺少评测材料"])
        ],
    )
    judgement = DimensionJudgement(
        evidence_status="missing",
        score=0,
        confidence="unavailable",
        evidence_refs=[],
        missing_items=["缺少评测材料"],
        conflicts=[],
        rationale="当前材料无法评分",
    )

    with pytest.raises(ValueError, match="missing"):
        validate_dimension_judgement(branch, judgement)


def test_receive_input_rejects_incomplete_candidate_dimension_matrix():
    import pytest

    from app.talent_evaluation_report import _receive_input

    dimensions = [
        _dimension("rag_delivery", 60).model_dump(mode="json"),
        _dimension("agent_evaluation", 40).model_dump(mode="json"),
    ]
    branch_results = [
        {
            "task_id": "C001:1",
            "candidate_id": "C001",
            "dimension_number": 1,
            "execution_status": "succeeded",
            "requirements": [
                _branch_requirement(missing_information=["缺少项目材料"])
            ],
            "tool_call_ids": [],
        }
    ]

    with pytest.raises(ValueError, match="C001:2"):
        _receive_input(
            {
                "dimensions": dimensions,
                "branch_results": branch_results,
                "evidence_items": [],
            }
        )


def test_report_graph_scores_aggregates_ranks_and_renders_trusted_citations():
    from app.talent_evaluation_report import (
        CandidateReportDraft,
        ConsistencyIssue,
        ConsistencyReview,
        DimensionJudgement,
        ReportDraft,
        ReportPoint,
        build_talent_evaluation_report_graph,
    )

    dimensions = [
        _dimension("rag_delivery", 60).model_dump(mode="json"),
        _dimension("agent_evaluation", 40).model_dump(mode="json"),
    ]

    def requirement(chunk_id, quote, source_label, missing_information=None):
        return {
            "requirement_id": "requirement:1",
            "query": "项目职责与交付结果",
            "status": "partial" if missing_information else "sufficient",
            "reason": "evidence_review",
            "extraction_status": "succeeded",
            "facts": [
                {
                    "event": "项目",
                    "period": "2025",
                    "claim": "具备相关经历",
                    "answer": "yes",
                    "sources": [
                        {
                            "citation_id": chunk_id,
                            "chunk_id": chunk_id,
                            "quote": quote,
                            "quote_start": 0,
                            "quote_end": len(quote),
                            "source_label": source_label,
                        }
                    ],
                }
            ],
            "conflicts": [],
            "missing_information": missing_information or [],
            "citations": [],
        }

    branch_results = [
        {
            "task_id": "C001:1",
            "candidate_id": "C001",
            "dimension_number": 1,
            "execution_status": "succeeded",
            "requirements": [
                requirement("chunk-1", "主导企业知识库 RAG 上线", "项目复盘")
            ],
            "tool_call_ids": ["T1"],
        },
        {
            "task_id": "C001:2",
            "candidate_id": "C001",
            "dimension_number": 2,
            "execution_status": "succeeded",
            "requirements": [
                requirement(
                    "chunk-2",
                    "建立了 Agent 评测集",
                    "面试记录",
                    ["缺少完整回归记录"],
                )
            ],
            "tool_call_ids": ["T2"],
        },
        {
            "task_id": "C002:1",
            "candidate_id": "C002",
            "dimension_number": 1,
            "execution_status": "succeeded",
            "requirements": [
                requirement(
                    "chunk-3",
                    "参与 RAG 检索模块开发",
                    "简历",
                    ["缺少量化结果"],
                )
            ],
            "tool_call_ids": ["T3"],
        },
        {
            "task_id": "C002:2",
            "candidate_id": "C002",
            "dimension_number": 2,
            "execution_status": "failed",
            "requirements": [],
            "tool_call_ids": [],
            "error_code": "branch_timeout",
            "error_message": "证据检索超时",
        },
    ]
    scores = {
        "C001:1": DimensionJudgement(
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=["chunk-1"],
            rationale="完整主导并交付 RAG 项目",
        ),
        "C001:2": DimensionJudgement(
            evidence_status="partial",
            score=3,
            confidence="medium",
            evidence_refs=["chunk-2"],
            missing_items=["缺少完整回归记录"],
            rationale="已有评测集事实",
        ),
        "C002:1": DimensionJudgement(
            evidence_status="partial",
            score=3,
            confidence="medium",
            evidence_refs=["chunk-3"],
            missing_items=["缺少量化结果"],
            rationale="已有参与事实",
        ),
    }
    scored_task_ids = []

    def scorer(dimension, branch, evidence):
        scored_task_ids.append(branch.task_id)
        from app.talent_evaluation_report import branch_evidence_refs

        assert {item.ref_id for item in evidence} == set(branch_evidence_refs(branch))
        return scores[branch.task_id]

    def report_writer(candidates, evidence):
        assert [item.candidate_id for item in candidates] == ["C001", "C002"]
        assert candidates[0].consistency_issues[0].dimension_numbers == [
            1,
            2,
        ]
        return ReportDraft(
            title="人才评估与推荐报告",
            overview="C001 已确认匹配度较高",
            candidates=[
                CandidateReportDraft(
                    candidate_id="C001",
                    summary="RAG 交付经验完整",
                    highlights=[ReportPoint(text="主导 RAG 项目上线", evidence_refs=["chunk-1"])],
                    risks=["需补充 Agent 回归记录"],
                ),
                CandidateReportDraft(
                    candidate_id="C002",
                    summary="当前材料覆盖不足",
                    highlights=[ReportPoint(text="参与 RAG 模块开发", evidence_refs=["chunk-3"])],
                    risks=["Agent 评测分支执行失败"],
                ),
            ],
        )

    def consistency_reviewer(assessments, evidence):
        assert len(assessments) == 2
        if assessments[0].candidate_id == "C002":
            return ConsistencyReview()
        return ConsistencyReview(
            issues=[
                ConsistencyIssue(
                    candidate_id="C001",
                    dimension_numbers=[1, 2],
                    evidence_refs=["chunk-1", "chunk-2"],
                    message="项目时间范围需要进一步核对",
                )
            ]
        )

    graph = build_talent_evaluation_report_graph(
        dimension_scorer=scorer,
        consistency_reviewer=consistency_reviewer,
        report_writer=report_writer,
    )
    result = graph.invoke(
        {
            "dimensions": dimensions,
            "branch_results": branch_results,
        }
    )

    assert set(scored_task_ids) == {
        "C001:1",
        "C001:2",
        "C002:1",
    }
    assert result["ranking"] == ["C001", "C002"]
    assert result["candidate_assessments"][0]["confirmed_score"] == 84.0
    assert result["candidate_assessments"][0]["confidence"] == "low"
    assert result["candidate_assessments"][1]["confirmed_score"] == 36.0
    assert result["candidate_assessments"][1]["dimensions"][1]["evidence_status"] == "failed"
    assert "[E1](/api/evidence/citations/chunk-1?quote_start=0&quote_end=14)" in result["report"]
    assert result["status"] == "completed"


def test_validate_report_draft_rejects_evidence_from_another_candidate():
    import pytest

    from app.talent_evaluation_report import (
        CandidateAssessment,
        CandidateReportDraft,
        DimensionAssessment,
        EvidenceItem,
        ReportDraft,
        ReportPoint,
        validate_report_draft,
    )

    candidates = [
        CandidateAssessment(
            candidate_id="C001",
            confirmed_score=100,
            possible_score=100,
            evidence_coverage_percent=100,
            confidence="high",
            dimensions=[
                DimensionAssessment(
                    task_id="C001:1",
                    candidate_id="C001",
                    dimension_number=1,
                    evidence_status="sufficient",
                    score=5,
                    confidence="high",
                    evidence_refs=["E1"],
                    rationale="有完整交付事实",
                )
            ],
        )
    ]
    evidence_items = [
        EvidenceItem(
            ref_id="E1",
            candidate_id="C002",
            quote="其他候选人证据",
            source_label="简历",
            citation_url="/api/evidence/citations/chunk-2",
        )
    ]
    draft = ReportDraft(
        title="报告",
        overview="评估结果",
        candidates=[
            CandidateReportDraft(
                candidate_id="C001",
                summary="综合表现良好",
                highlights=[ReportPoint(text="RAG 经验完整", evidence_refs=["E1"])],
            )
        ],
    )

    with pytest.raises(ValueError, match="E1"):
        validate_report_draft(draft, candidates, evidence_items)


def test_structured_dimension_scorer_uses_anchors_and_evidence_without_counter_fields():
    from app.talent_evaluation_dispatch import BranchEvidenceDraft
    from app.talent_evaluation_report import (
        DimensionJudgement,
        EvidenceItem,
        build_structured_dimension_scorer,
    )

    expected = DimensionJudgement(
        evidence_status="sufficient",
        score=5,
        confidence="high",
        evidence_refs=["E1"],
        rationale="完整主导并交付 RAG 项目",
    )

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is DimensionJudgement
            return self

        def invoke(self, messages):
            self.messages = messages
            return expected

    model = StructuredModel()
    scorer = build_structured_dimension_scorer(lambda: model)
    result = scorer(
        _dimension("rag_delivery", 100),
        BranchEvidenceDraft(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            execution_status="succeeded",
            requirements=[_branch_requirement("E1")],
        ),
        [
            EvidenceItem(
                ref_id="E1",
                candidate_id="C001",
                quote="主导 RAG 项目并完成上线",
                source_label="项目复盘",
                citation_url="/api/evidence/citations/chunk-1",
            )
        ],
    )

    assert result == expected
    assert "score_anchors" in model.messages[1][1]
    assert "主导 RAG 项目" in model.messages[1][1]


def test_structured_dimension_scorer_uses_short_evidence_refs_and_restores_chunk_ids():
    from app.talent_evaluation_dispatch import BranchEvidenceDraft
    from app.talent_evaluation_report import (
        DimensionJudgement,
        EvidenceItem,
        build_structured_dimension_scorer,
    )

    chunk_id = "174b7b02-20dc-49e9-ab2c-bffe90eb4760"

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is DimensionJudgement
            return self

        def invoke(self, messages):
            self.messages = messages
            return DimensionJudgement(
                evidence_status="sufficient",
                score=5,
                confidence="high",
                evidence_refs=["E1"],
                rationale="证据完整",
            )

    model = StructuredModel()
    scorer = build_structured_dimension_scorer(lambda: model)
    result = scorer(
        _dimension("rag_delivery", 100),
        BranchEvidenceDraft(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            execution_status="succeeded",
            requirements=[_branch_requirement(chunk_id)],
        ),
        [
            EvidenceItem(
                ref_id=chunk_id,
                model_ref="E1",
                candidate_id="C001",
                quote="主导 RAG 项目并完成上线",
                source_label="项目复盘",
                citation_url=f"/api/evidence/citations/{chunk_id}",
            )
        ],
    )

    assert result.evidence_refs == [chunk_id]
    assert chunk_id not in model.messages[1][1]
    assert '"ref_id": "E1"' in model.messages[1][1]


def test_structured_report_writer_receives_ranked_assessments_and_verified_evidence():
    from app.talent_evaluation_report import (
        CandidateAssessment,
        CandidateReportDraft,
        DimensionAssessment,
        EvidenceItem,
        ReportDraft,
        ReportPoint,
        build_structured_report_writer,
    )

    expected = ReportDraft(
        title="人才评估与推荐报告",
        overview="C001 的已确认得分较高",
        candidates=[
            CandidateReportDraft(
                candidate_id="C001",
                summary="RAG 交付证据完整",
                highlights=[ReportPoint(text="主导 RAG 上线", evidence_refs=["E1"])],
                risks=[],
            )
        ],
    )

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is ReportDraft
            return self

        def invoke(self, messages):
            self.messages = messages
            return expected

    model = StructuredModel()
    writer = build_structured_report_writer(lambda: model)
    draft = writer(
        [
            CandidateAssessment(
                candidate_id="C001",
                confirmed_score=100,
                possible_score=100,
                evidence_coverage_percent=100,
                confidence="high",
                dimensions=[
                    DimensionAssessment(
                        task_id="C001:1",
                        candidate_id="C001",
                        dimension_number=1,
                        evidence_status="sufficient",
                        score=5,
                        confidence="high",
                        evidence_refs=["E1"],
                        rationale="完整主导并交付 RAG 项目",
                    )
                ],
            )
        ],
        [
            EvidenceItem(
                ref_id="E1",
                candidate_id="C001",
                quote="主导 RAG 项目并完成上线",
                source_label="项目复盘",
                citation_url="/api/evidence/citations/chunk-1",
            )
        ],
    )

    assert draft == expected
    assert "confirmed_score" in model.messages[1][1]
    assert "citation_url" not in model.messages[1][1]


def test_structured_report_writer_restores_short_refs_before_validation():
    from app.talent_evaluation_report import (
        CandidateAssessment,
        CandidateReportDraft,
        DimensionAssessment,
        EvidenceItem,
        ReportDraft,
        ReportPoint,
        build_structured_report_writer,
    )

    chunk_id = "174b7b02-20dc-49e9-ab2c-bffe90eb4760"

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is ReportDraft
            return self

        def invoke(self, messages):
            self.messages = messages
            return ReportDraft(
                title="人才评估报告",
                overview="候选人具备完整交付证据",
                candidates=[
                    CandidateReportDraft(
                        candidate_id="C001",
                        summary="RAG 交付证据完整",
                        highlights=[
                            ReportPoint(text="主导项目上线", evidence_refs=["E1"])
                        ],
                    )
                ],
            )

    candidate = CandidateAssessment(
        candidate_id="C001",
        confirmed_score=100,
        possible_score=100,
        evidence_coverage_percent=100,
        confidence="high",
        dimensions=[
            DimensionAssessment(
                task_id="C001:1",
                candidate_id="C001",
                dimension_number=1,
                evidence_status="sufficient",
                score=5,
                confidence="high",
                evidence_refs=[chunk_id],
                rationale="完整主导并交付 RAG 项目",
            )
        ],
    )
    evidence = EvidenceItem(
        ref_id=chunk_id,
        model_ref="E1",
        candidate_id="C001",
        quote="主导 RAG 项目并完成上线",
        source_label="项目复盘",
        citation_url=f"/api/evidence/citations/{chunk_id}",
    )
    model = StructuredModel()

    draft = build_structured_report_writer(lambda: model)([candidate], [evidence])

    assert draft.candidates[0].highlights[0].evidence_refs == [chunk_id]
    assert chunk_id not in model.messages[1][1]
    assert '"ref_id": "E1"' in model.messages[1][1]


def test_structured_output_models_describe_every_model_generated_field():
    from app.talent_evaluation_report import (
        CandidateReportDraft,
        ConsistencyIssue,
        ConsistencyReview,
        DimensionJudgement,
        ReportDraft,
        ReportPoint,
    )

    for schema in (
        DimensionJudgement,
        ConsistencyIssue,
        ConsistencyReview,
        ReportPoint,
        CandidateReportDraft,
        ReportDraft,
    ):
        properties = schema.model_json_schema()["properties"]
        assert all(field.get("description") for field in properties.values()), schema


def test_consistency_issue_lowers_confidence_without_rewriting_scores():
    from app.talent_evaluation_report import (
        ConsistencyIssue,
        DimensionAssessment,
        aggregate_candidate_assessments,
    )

    dimensions = [
        _dimension("rag_delivery", 60),
        _dimension("collaboration_delivery", 40),
    ]
    assessments = [
        DimensionAssessment(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=["E1"],
            rationale="主导 RAG 方案",
        ),
        DimensionAssessment(
            task_id="C001:2",
            candidate_id="C001",
            dimension_number=2,
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=["E2"],
            rationale="负责跨团队交付",
        ),
    ]
    issue = ConsistencyIssue(
        candidate_id="C001",
        dimension_numbers=[1, 2],
        evidence_refs=["E1", "E2"],
        message="两份材料对主导职责的描述不一致",
    )

    result = aggregate_candidate_assessments(dimensions, assessments, [issue])

    assert result[0].confirmed_score == 100.0
    assert result[0].possible_score == 100.0
    assert result[0].confidence == "low"
    assert result[0].consistency_issues == [issue]


def test_structured_consistency_reviewer_uses_only_scored_evidence():
    from app.talent_evaluation_report import (
        ConsistencyIssue,
        ConsistencyReview,
        DimensionAssessment,
        EvidenceItem,
        build_structured_consistency_reviewer,
    )

    expected = ConsistencyReview(
        issues=[
            ConsistencyIssue(
                candidate_id="C001",
                dimension_numbers=[1, 2],
                evidence_refs=["E1", "E2"],
                message="两个维度对项目主导职责的描述不一致",
            )
        ]
    )

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is ConsistencyReview
            return self

        def invoke(self, messages):
            self.messages = messages
            return expected

    model = StructuredModel()
    reviewer = build_structured_consistency_reviewer(lambda: model)
    assessments = [
        DimensionAssessment(
            task_id="C001:1",
            candidate_id="C001",
            dimension_number=1,
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=["E1"],
            rationale="主导 RAG 方案",
        ),
        DimensionAssessment(
            task_id="C001:2",
            candidate_id="C001",
            dimension_number=2,
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=["E2"],
            rationale="负责跨团队交付",
        ),
    ]
    evidence = [
        EvidenceItem(
            ref_id="E1",
            candidate_id="C001",
            quote="候选人主导 RAG 方案",
            source_label="项目复盘",
            citation_url="/api/evidence/citations/chunk-1",
        ),
        EvidenceItem(
            ref_id="E2",
            candidate_id="C001",
            quote="候选人作为参与者配合交付",
            source_label="面试记录",
            citation_url="/api/evidence/citations/chunk-2",
        ),
    ]

    result = reviewer(assessments, evidence)

    assert result == expected
    assert "citation_url" not in model.messages[1][1]
    assert "E1" in model.messages[1][1]


def test_structured_consistency_reviewer_restores_short_refs():
    from app.talent_evaluation_report import (
        ConsistencyIssue,
        ConsistencyReview,
        DimensionAssessment,
        EvidenceItem,
        build_structured_consistency_reviewer,
    )

    chunk_ids = [
        "174b7b02-20dc-49e9-ab2c-bffe90eb4760",
        "a5096790-dc7a-43b5-b977-8d238c5122b7",
    ]

    class StructuredModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is ConsistencyReview
            return self

        def invoke(self, messages):
            self.messages = messages
            return ConsistencyReview(
                issues=[
                    ConsistencyIssue(
                        candidate_id="C001",
                        dimension_numbers=[1, 2],
                        evidence_refs=["E1", "E2"],
                        message="职责描述不一致",
                    )
                ]
            )

    assessments = [
        DimensionAssessment(
            task_id=f"C001:{number}",
            candidate_id="C001",
            dimension_number=number,
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=[chunk_id],
            rationale="存在可核验证据",
        )
        for number, chunk_id in enumerate(chunk_ids, start=1)
    ]
    evidence = [
        EvidenceItem(
            ref_id=chunk_id,
            model_ref=f"E{number}",
            candidate_id="C001",
            quote="项目职责描述",
            source_label="项目材料",
            citation_url=f"/api/evidence/citations/{chunk_id}",
        )
        for number, chunk_id in enumerate(chunk_ids, start=1)
    ]
    model = StructuredModel()

    review = build_structured_consistency_reviewer(lambda: model)(
        assessments,
        evidence,
    )

    assert review.issues[0].evidence_refs == chunk_ids
    assert all(chunk_id not in model.messages[1][1] for chunk_id in chunk_ids)


def test_score_dimensions_runs_twelve_model_calls_concurrently():
    import threading
    import time

    from app.talent_evaluation_report import (
        DimensionJudgement,
        _score_dimensions,
    )

    active = 0
    max_active = 0
    lock = threading.Lock()

    def scorer(dimension, branch, evidence_items):
        nonlocal active, max_active
        del dimension, evidence_items
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return DimensionJudgement(
            evidence_status="sufficient",
            score=5,
            confidence="high",
            evidence_refs=[f"chunk-{branch.candidate_id}"],
            rationale="存在完整交付证据",
        )

    branch_results = []
    evidence_items = []
    for number in range(12):
        candidate_id = f"C{number:03d}"
        chunk_id = f"chunk-{candidate_id}"
        branch_results.append(
            {
                "task_id": f"{candidate_id}:1",
                "candidate_id": candidate_id,
                "dimension_number": 1,
                "execution_status": "succeeded",
                "requirements": [_branch_requirement(chunk_id)],
            }
        )
        evidence_items.append(
            {
                "ref_id": chunk_id,
                "model_ref": f"E{number + 1}",
                "candidate_id": candidate_id,
                "quote": "主导项目上线",
                "source_label": "项目复盘",
                "citation_url": f"/api/evidence/citations/{chunk_id}",
            }
        )

    result = _score_dimensions(scorer, max_concurrency=12)(
        {
            "dimensions": [_dimension("rag_delivery", 100).model_dump(mode="json")],
            "branch_results": branch_results,
            "evidence_items": evidence_items,
        }
    )

    assert max_active == 12
    assert [item["task_id"] for item in result["dimension_assessments"]] == [
        item["task_id"] for item in branch_results
    ]


def test_consistency_review_runs_once_per_candidate_concurrently():
    import threading
    import time

    from app.talent_evaluation_report import (
        ConsistencyReview,
        _review_consistency,
    )

    active = 0
    max_active = 0
    reviewed_candidates = []
    lock = threading.Lock()

    def reviewer(assessments, evidence_items):
        nonlocal active, max_active
        del evidence_items
        candidate_ids = {item.candidate_id for item in assessments}
        with lock:
            reviewed_candidates.append(candidate_ids)
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return ConsistencyReview()

    assessments = []
    for candidate_id in ("C001", "C002"):
        for dimension_number in (1, 2):
            assessments.append(
                {
                    "task_id": f"{candidate_id}:{dimension_number}",
                    "candidate_id": candidate_id,
                    "dimension_number": dimension_number,
                    "evidence_status": "missing",
                    "score": None,
                    "confidence": "unavailable",
                    "evidence_refs": [],
                    "missing_items": ["缺少材料"],
                    "conflicts": [],
                    "rationale": "当前材料无法评分",
                }
            )

    result = _review_consistency(reviewer, max_concurrency=12)(
        {
            "dimension_assessments": assessments,
            "evidence_items": [],
        }
    )

    assert max_active == 2
    assert {frozenset(item) for item in reviewed_candidates} == {
        frozenset({"C001"}),
        frozenset({"C002"}),
    }
    assert result["consistency_issues"] == []


def test_verify_script_reuses_report_runtime_graph_and_dispatch_output(
    monkeypatch,
    tmp_path,
):
    import importlib
    import json
    import sys
    from types import ModuleType

    class RecordingGraph:
        def __init__(self):
            self.input = None

        def invoke(self, input_state):
            self.input = input_state
            return {"status": "completed", "ranking": []}

    graph = RecordingGraph()
    runtime = ModuleType("app.talent_evaluation_runtime")
    runtime.report_graph = graph
    monkeypatch.setitem(sys.modules, "app.talent_evaluation_runtime", runtime)
    sys.modules.pop("scripts.verify_talent_evaluation_report", None)
    input_path = tmp_path / "dispatch.json"
    input_path.write_text(
        json.dumps(
            {
                "dimensions": [{"name": "RAG", "weight_percent": 100}],
                "branch_results": [{"task_id": "C001:1"}],
                "candidate_ids": ["C001"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verifier = importlib.import_module("scripts.verify_talent_evaluation_report")
    result = verifier.run_verification(input_path)

    assert verifier.graph is runtime.report_graph
    assert graph.input == {
        "dimensions": [{"name": "RAG", "weight_percent": 100}],
        "branch_results": [{"task_id": "C001:1"}],
    }
    assert result == {"status": "completed", "ranking": []}
    assert not hasattr(verifier, "build_graph")
    assert not hasattr(verifier, "JUDGEMENTS")
    sys.modules.pop("scripts.verify_talent_evaluation_report", None)


def test_report_runtime_builds_graph_with_structured_model_components(monkeypatch):
    import importlib
    import sys

    from app import evidence_index_service, model_provider, reranker
    from app import talent_evaluation_dispatch as dispatch_module
    from app import talent_evaluation_report as report_module

    model = object()
    dispatch_graph = object()
    report_graph = object()
    report_kwargs = {}

    monkeypatch.setattr(model_provider, "get_chat_model", lambda temperature=0: model)
    monkeypatch.setattr(model_provider, "get_embedding_model", lambda: object())
    monkeypatch.setattr(reranker, "get_reranker", lambda: object())
    monkeypatch.setattr(evidence_index_service, "get_evidence_store", lambda: object())
    monkeypatch.setattr(
        dispatch_module,
        "build_talent_evaluation_dispatch_graph",
        lambda **kwargs: dispatch_graph,
    )

    def build_report(**kwargs):
        report_kwargs.update(kwargs)
        return report_graph

    monkeypatch.setattr(
        report_module,
        "build_talent_evaluation_report_graph",
        build_report,
    )
    sys.modules.pop("app.talent_evaluation_runtime", None)

    runtime = importlib.import_module("app.talent_evaluation_runtime")

    assert runtime.graph is dispatch_graph
    assert runtime.report_graph is report_graph
    assert set(report_kwargs) == {
        "dimension_scorer",
        "consistency_reviewer",
        "report_writer",
        "max_scoring_concurrency",
        "max_consistency_concurrency",
    }
    assert callable(report_kwargs["dimension_scorer"])
    assert callable(report_kwargs["consistency_reviewer"])
    assert callable(report_kwargs["report_writer"])
    assert report_kwargs["max_scoring_concurrency"] == 12
    assert report_kwargs["max_consistency_concurrency"] == 12
    sys.modules.pop("app.talent_evaluation_runtime", None)


def test_langgraph_config_registers_report_runtime_graph():
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).resolve().parents[1] / "langgraph.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["graphs"]["talent_evaluation_report"] == (
        "./app/talent_evaluation_runtime.py:report_graph"
    )


def test_report_graph_derives_evidence_items_and_missing_information_from_branch_packs():
    from app.talent_evaluation_report import (
        CandidateReportDraft,
        DimensionJudgement,
        ReportDraft,
        ReportPoint,
        build_talent_evaluation_report_graph,
    )

    dimensions = [_dimension("rag_delivery", 100).model_dump(mode="json")]
    branch_results = [
        {
            "task_id": "C001:1",
            "candidate_id": "C001",
            "dimension_number": 1,
            "execution_status": "succeeded",
            "requirements": [
                {
                    "requirement_id": "1:1",
                    "query": "RAG 项目职责",
                    "status": "partial",
                    "reason": "evidence_review",
                    "extraction_status": "succeeded",
                    "facts": [
                        {
                            "event": "星河项目",
                            "period": "2025",
                            "claim": "负责 RAG 方案",
                            "answer": "yes",
                            "sources": [
                                {
                                    "citation_id": "citation-1",
                                    "chunk_id": "chunk-1",
                                    "quote": "主导 RAG 方案上线",
                                    "quote_start": 0,
                                    "quote_end": 10,
                                    "source_label": "项目复盘",
                                }
                            ],
                        }
                    ],
                    "conflicts": [],
                    "missing_information": ["缺少量化结果"],
                    "citations": [],
                }
            ],
            "tool_call_ids": ["call-1"],
        }
    ]

    def scorer(dimension, branch, evidence_items):
        assert [item.ref_id for item in evidence_items] == ["chunk-1"]
        assert evidence_items[0].source_label == "项目复盘"
        assert branch.requirements[0].missing_information == ["缺少量化结果"]
        return DimensionJudgement(
            evidence_status="partial",
            score=3,
            confidence="medium",
            evidence_refs=["chunk-1"],
            missing_items=[],
            rationale="已有项目职责事实",
        )

    def writer(candidates, evidence_items):
        assert candidates[0].dimensions[0].missing_items == ["缺少量化结果"]
        assert [item.ref_id for item in evidence_items] == ["chunk-1"]
        return ReportDraft(
            title="人才评估报告",
            overview="C001 有部分可验证证据",
            candidates=[
                CandidateReportDraft(
                    candidate_id="C001",
                    summary="存在 RAG 项目事实",
                    highlights=[
                        ReportPoint(text="主导 RAG 方案上线", evidence_refs=["chunk-1"])
                    ],
                    risks=["缺少量化结果"],
                )
            ],
        )

    graph = build_talent_evaluation_report_graph(
        dimension_scorer=scorer,
        report_writer=writer,
    )
    result = graph.invoke(
        {
            "dimensions": dimensions,
            "branch_results": branch_results,
        }
    )

    assert result["evidence_items"][0]["ref_id"] == "chunk-1"
    assert "[E1](/api/evidence/citations/chunk-1?quote_start=0&quote_end=10)" in result["report"]
