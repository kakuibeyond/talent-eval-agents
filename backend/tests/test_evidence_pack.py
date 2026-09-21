from copy import deepcopy
import importlib.util
import pytest


def module():
    assert importlib.util.find_spec('app.evidence_pack'), '第 9 课证据整理模块尚未实现'
    from app import evidence_pack
    return evidence_pack


def source(cid='a', candidate='C001'):
    return {'chunk_id': cid, 'candidate_id': candidate, 'content': '负责星河项目。',
            'citation_id': cid, 'document_version_id': 'v1', 'requirement_ids': ['S1']}


def extraction():
    return {'facts': [{'event': '星河项目', 'period': '2025',
        'claim': '担任项目负责人', 'answer': 'yes',
        'sources': [{'chunk_id': 'a', 'quote': '负责星河项目'}]}],
        'fully_supported': True, 'missing_information': []}


def build(data=None, sources=None, candidates=None):
    return module().build_evidence_packs(
        candidate_ids=candidates or ['C001'],
        requirements=[{'requirement_id': 'S1', 'query': '负责过星河项目'}],
        sources=sources if sources is not None else [source()],
        extract=lambda requirement, rows: data if data is not None else extraction())


def test_duplicate_chunk_is_not_counted_twice_and_missing_candidate_is_preserved():
    packs = build(sources=[source(), source()], candidates=['C001', 'C002'])
    assert packs[0]['schema_version'] == '2.0'
    assert len(packs[0]['requirements'][0]['citations']) == 1
    assert packs[0]['requirements'][0]['status'] == 'sufficient'
    assert packs[1]['requirements'][0]['status'] == 'missing'


def test_same_fact_merges_sources_without_dropping_provenance():
    data = extraction()
    second = deepcopy(data['facts'][0])
    second['sources'][0]['chunk_id'] = 'b'
    data['facts'].append(second)
    requirement = build(data, [source(), source('b')])[0]['requirements'][0]
    assert len(requirement['facts']) == 1
    assert len(requirement['facts'][0]['sources']) == 2


def test_verified_quote_includes_offsets_from_chunk_content():
    reference = build()[0]['requirements'][0]['facts'][0]['sources'][0]
    assert reference['quote'] == '负责星河项目'
    assert reference['quote_start'] == 0
    assert reference['quote_end'] == 6


def test_quote_with_outer_whitespace_is_trimmed_before_exact_source_validation():
    data = extraction()
    data['facts'][0]['sources'][0]['quote'] = '  负责星河项目  '

    reference = build(data)[0]['requirements'][0]['facts'][0]['sources'][0]

    assert reference['quote'] == '负责星河项目'
    assert reference['quote_start'] == 0
    assert reference['quote_end'] == 6


def test_unverifiable_quote_logs_safe_diagnostic_context(caplog):
    data = extraction()
    data['facts'][0]['sources'][0]['quote'] = '主持千人团队'

    with caplog.at_level('ERROR', logger='app.evidence_pack'):
        result = build(data)[0]['requirements'][0]

    assert result['extraction_status'] == 'failed'
    message = caplog.messages[-1]
    assert 'event=evidence_extraction_failed' in message
    assert 'function=_validate_fact_sources' in message
    assert 'stage=source_validation' in message
    assert 'error_code=quote_not_found' in message
    assert 'candidate_id=C001' in message
    assert 'requirement_id=S1' in message
    assert 'chunk_id=a' in message
    assert 'source_count=1' in message
    assert '负责星河项目。' not in message
    assert '主持千人团队' not in message


@pytest.mark.parametrize('bad', ['unknown_id', 'fabricated_quote'])
def test_unverifiable_model_output_falls_back_to_raw_citations(bad):
    data = extraction()
    data['facts'][0]['sources'][0].update(
        {'chunk_id': 'other-person'} if bad == 'unknown_id' else {'quote': '主持千人团队'})
    result = build(data)[0]['requirements'][0]
    assert result['extraction_status'] == 'failed'
    assert result['facts'] == []
    assert len(result['citations']) == 1
    assert result['status'] == 'partial'


def test_yes_and_no_answers_in_same_scope_form_conflict():
    data = extraction()
    opposite = deepcopy(data['facts'][0])
    opposite.update(answer='no')
    opposite['sources'] = [{'chunk_id': 'b', 'quote': '仅参与星河项目'}]
    data['facts'].append(opposite)
    rows = [source(), dict(source('b'), content='仅参与星河项目。')]
    requirement = build(data, rows)[0]['requirements'][0]
    assert requirement['status'] == 'conflicting'
    assert requirement['reason'] == 'evidence_review'
    assert requirement['conflicts'] == [[0, 1]]


def test_yes_and_no_answers_in_different_periods_are_not_a_conflict():
    data = extraction()
    opposite = deepcopy(data['facts'][0])
    opposite.update(period='2024', answer='no')
    opposite['sources'] = [{'chunk_id': 'b', 'quote': '仅参与星河项目'}]
    data['facts'].append(opposite)
    data['fully_supported'] = False
    rows = [source(), dict(source('b'), content='仅参与星河项目。')]
    requirement = build(data, rows)[0]['requirements'][0]
    assert requirement['status'] == 'partial'
    assert requirement['conflicts'] == []


def test_noncanonical_period_falls_back_instead_of_entering_conflict_check():
    data = extraction()
    data['facts'][0]['period'] = '2025 年 1 月至 2025 年 6 月'
    result = build(data)[0]['requirements'][0]
    assert result['status'] == 'partial'
    assert result['extraction_status'] == 'failed'


def test_model_failure_is_not_material_absence():
    def fail(*args):
        raise TimeoutError('private details')
    packs = module().build_evidence_packs(candidate_ids=['C001'], requirements=[
        {'requirement_id': 'S1', 'query': '负责过星河项目'}], sources=[source()], extract=fail)
    result = packs[0]['requirements'][0]
    assert result['status'] == 'partial'
    assert result['extraction_status'] == 'failed'
    assert 'private details' not in str(result)


def test_irrelevant_evidence_is_distinguished_from_no_hits():
    result = build({'facts': [], 'fully_supported': False,
        'missing_information': ['没有项目职责信息']})[0]['requirements'][0]
    assert result['status'] == 'missing'
    assert result['reason'] == 'no_relevant_evidence'


def test_blank_missing_information_is_rejected():
    data = extraction()
    data['missing_information'] = ['']
    result = build(data)[0]['requirements'][0]
    assert result['status'] == 'partial'
    assert result['extraction_status'] == 'failed'


def test_model_extractor_uses_json_mode_and_sends_the_output_schema():
    calls = {}

    class StructuredModel:
        def invoke(self, messages):
            calls['messages'] = messages
            return module().EvidenceExtraction(
                facts=[],
                fully_supported=False,
                missing_information=['材料未说明职责'],
            )

    class Model:
        def with_structured_output(self, schema, **kwargs):
            calls['schema'] = schema
            calls['kwargs'] = kwargs
            return StructuredModel()

    extract = module().model_extractor(Model())
    result = extract(
        {'requirement_id': 'S1', 'query': '负责过星河项目'},
        [source()],
    )

    payload = calls['messages'][1][1]
    assert result.missing_information == ['材料未说明职责']
    assert calls['schema'] is module().EvidenceExtraction
    assert calls['kwargs'] == {'method': 'json_mode'}
    assert 'output_schema' in payload


def test_model_extractor_uses_short_source_refs_and_restores_real_chunk_ids():
    calls = {}

    class StructuredModel:
        def invoke(self, messages):
            calls['payload'] = __import__('json').loads(messages[1][1])
            return module().EvidenceExtraction.model_validate({
                'facts': [{
                    'event': '星河项目',
                    'period': '2025',
                    'claim': '担任项目负责人',
                    'answer': 'yes',
                    'sources': [{
                        'chunk_id': 'source_1',
                        'quote': '负责星河项目',
                    }],
                }],
                'fully_supported': True,
                'missing_information': [],
            })

    class Model:
        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return StructuredModel()

    result = module().model_extractor(Model())(
        {'requirement_id': 'S1', 'query': '负责过星河项目'},
        [source('a-real-uuid')],
    )

    assert calls['payload']['sources'][0]['chunk_id'] == 'source_1'
    assert result.facts[0].sources[0].chunk_id == 'a-real-uuid'
