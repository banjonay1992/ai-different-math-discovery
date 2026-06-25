"""Connected-vs-isolated science campaign benefit evaluator.

This module compares symbolic campaign management with isolated AI Different
evidence versus connected module-family evidence. It does not run science
experiments, import sibling projects, or claim project-owned checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .campaign_outcome_assessor import CAMPAIGN_OUTCOME_LEDGER_KIND
from .campaign_planner import CAMPAIGN_LEDGER_KIND
from .cross_module_adjudicator import ADJUDICATOR_LEDGER_KIND
from .evidence_scorecard import SCORECARD_LEDGER_KIND
from .experiment_agenda import AGENDA_LEDGER_KIND
from .experiment_contracts import CONTRACT_LEDGER_KIND, validate_evaluator_ledger
from .family_outcome_evaluator import EVALUATOR_LEDGER_KIND
from .hypothesis_lifecycle import LIFECYCLE_LEDGER_KIND
from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)


SCIENCE_BENEFIT_LEDGER_KIND = 'ai_different.science_campaign_benefit_ledger'
GATE_ORDER = ('math_proof', 'code_proof', 'language_epoch_plan')
CONNECTED_SENDERS = {'funfun', 'language_model_2', 'code_module'}


def empty_science_benefit_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_BENEFIT_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'evaluated_scenario_ids': [],
        'benefit_records': [],
        'scenario_records': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_benefit_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_benefit_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_benefit_ledger(ledger)


def write_science_benefit_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_benefit_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_benefit_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science benefit ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_BENEFIT_LEDGER_KIND:
        raise ValueError('science benefit ledger has wrong ledger_kind')
    for field in (
        'processed_message_ids',
        'processed_source_hashes',
        'evaluated_scenario_ids',
        'outgoing_response_ids',
    ):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('benefit_records', 'scenario_records'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science benefit latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_BENEFIT_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'evaluated_scenario_ids': _unique_strings(ledger['evaluated_scenario_ids']),
        'benefit_records': list(ledger['benefit_records']),
        'scenario_records': list(ledger['scenario_records']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_benefit_transcript(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {'path': None, 'messages': [], 'invalid_messages': []}
    transcript = Path(path)
    if not transcript.exists():
        return {'path': str(transcript), 'messages': [], 'invalid_messages': []}
    messages = []
    invalid = []
    with transcript.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                messages.append(validate_module_chat_message(json.loads(text)))
            except (json.JSONDecodeError, ValueError) as error:
                invalid.append({'line': line_number, 'error': str(error), 'raw': text})
    return {'path': str(transcript), 'messages': messages, 'invalid_messages': invalid}


def load_plain_json(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open('r', encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('plain science benefit input must be a JSON object')
    return value


def build_science_benefit_evaluation(
    *,
    transcript_messages: list[dict[str, Any]],
    benefit_ledger: dict[str, Any],
    evaluator_ledger: dict[str, Any] | None = None,
    outcome_ledger: dict[str, Any] | None = None,
    contract_ledger: dict[str, Any] | None = None,
    adjudicator_ledger: dict[str, Any] | None = None,
    agenda_ledger: dict[str, Any] | None = None,
    lifecycle_ledger: dict[str, Any] | None = None,
    scorecard_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    campaign_outcome_ledger: dict[str, Any] | None = None,
    prior_benefit_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    del runtime_memory_data
    ledger = validate_science_benefit_ledger(benefit_ledger)
    evaluator = _valid_evaluator_or_empty(evaluator_ledger or {})
    prior_outcome = _valid_prior_outcome_or_empty(outcome_ledger or {})
    contracts = _valid_contract_or_empty(contract_ledger or {})
    adjudicator = _valid_adjudicator_or_empty(adjudicator_ledger or {})
    agenda = _valid_agenda_or_empty(agenda_ledger or {})
    lifecycle = _valid_lifecycle_or_empty(lifecycle_ledger or {})
    scorecard = _valid_scorecard_or_empty(scorecard_ledger or {})
    campaign = _valid_campaign_or_empty(campaign_ledger or {})
    campaign_outcome = _valid_campaign_outcome_or_empty(campaign_outcome_ledger or {})
    prior_benefit = _valid_prior_benefit_or_empty(prior_benefit_ledger or {})
    source_hash = _source_hash(
        evaluator,
        prior_outcome,
        contracts,
        adjudicator,
        agenda,
        lifecycle,
        scorecard,
        campaign,
        campaign_outcome,
        prior_benefit,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) not in processed
    ]
    skipped_messages = [
        message for message in transcript_messages
        if module_chat_message_id(message) in processed
    ]
    scenarios = _extract_scenarios(
        ledger['scenario_records'],
        transcript_messages,
        campaign,
        campaign_outcome,
        scorecard,
        lifecycle,
        agenda,
        contracts,
    )
    if not new_messages and (not source_is_new or scenarios):
        selected = _noop_action('no new science benefit evidence or source ledger state')
    else:
        selected = _select_benefit_action(
            scenarios=scenarios,
            evaluated_ids=ledger['evaluated_scenario_ids'],
            project_owned_boundary=project_owned_boundary,
        )
    benefit_id = 'science_benefit_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'classification': selected['benefit_classification'],
        'scenario_id': selected.get('scenario_id'),
    })[:16]
    selected['benefit_id'] = benefit_id
    message = export_science_benefit_message(
        selected,
        scenarios=scenarios,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        source_hashes={
            'science_benefit_source_hash': source_hash,
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': prior_outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'campaign_outcome_ledger_hash': campaign_outcome.get('ledger_hash'),
            'prior_benefit_ledger_hash': prior_benefit.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    if selected['benefit_classification'] != 'summarize_noop' and selected.get('scenario_id'):
        ledger['evaluated_scenario_ids'] = _unique_strings(list(ledger['evaluated_scenario_ids']) + [selected['scenario_id']])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new and selected['benefit_classification'] in {
        'isolated_already_sufficient',
        'connected_adds_no_safe_benefit',
        'summarize_noop',
    }:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    status_counts = _benefit_status_counts(selected, scenarios)
    latest = {
        'benefit_id': benefit_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'status_counts': status_counts,
        'benefit_classification': selected['benefit_classification'],
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
    }
    record = {
        'benefit_id': benefit_id,
        'benefit_hash': stable_digest({'benefit_id': benefit_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'scenario_ids': [item.get('scenario_id') for item in scenarios],
        'hypothesis_ids': _unique_strings([item.get('hypothesis_id') for item in scenarios if item.get('hypothesis_id')]),
        'selected_scenario_id': selected.get('scenario_id'),
        'isolated_decision_summary': selected.get('isolated_decision_summary'),
        'connected_decision_summary': selected.get('connected_decision_summary'),
        'accepted_evidence_by_module': selected.get('accepted_evidence_by_module'),
        'missing_evidence_by_module': selected.get('missing_evidence_by_module'),
        'rejected_evidence_by_module': selected.get('rejected_evidence_by_module'),
        'theory_update_delta': selected.get('theory_update_delta'),
        'refinement_delta': selected.get('refinement_delta'),
        'retirement_delta': selected.get('retirement_delta'),
        'boundary_state': selected.get('boundary_state'),
        'selected_action': selected['selected_action'],
        'chosen_recipient': latest['chosen_recipient'],
        'outgoing_response_id': outgoing_id,
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'source_ledger_hashes': {
            'evaluator_ledger_hash': evaluator.get('ledger_hash'),
            'outcome_ledger_hash': prior_outcome.get('ledger_hash'),
            'contract_ledger_hash': contracts.get('ledger_hash'),
            'adjudicator_ledger_hash': adjudicator.get('ledger_hash'),
            'agenda_ledger_hash': agenda.get('ledger_hash'),
            'lifecycle_ledger_hash': lifecycle.get('ledger_hash'),
            'scorecard_ledger_hash': scorecard.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'campaign_outcome_ledger_hash': campaign_outcome.get('ledger_hash'),
            'prior_benefit_ledger_hash': prior_benefit.get('ledger_hash'),
        },
    }
    ledger['scenario_records'] = scenarios
    if new_messages or source_is_new or message is not None:
        ledger['benefit_records'] = list(ledger['benefit_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_benefit_message(
    selected: dict[str, Any],
    *,
    scenarios: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_campaign_benefit',
) -> dict[str, Any] | None:
    if selected['benefit_classification'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('chosen_recipient') or 'orchestrator')
    scenario = _find_scenario(scenarios, selected.get('scenario_id')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'scenario': scenario})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_campaign_benefit',
        'benefit_id': selected.get('benefit_id'),
        'scenario_id': selected.get('scenario_id'),
        'hypothesis_id': selected.get('hypothesis_id'),
        'benefit_classification': selected['benefit_classification'],
        'selected_action': selected['selected_action'],
        'isolated_vs_connected_comparison': {
            'isolated': selected.get('isolated_decision_summary'),
            'connected': selected.get('connected_decision_summary'),
        },
        'sibling_evidence_used': selected.get('sibling_evidence_used') or [],
        'theory_update_delta': selected.get('theory_update_delta'),
        'refinement_delta': selected.get('refinement_delta'),
        'retirement_delta': selected.get('retirement_delta'),
        'missing_evidence': selected.get('missing_evidence') or [],
        'boundary_notes': selected.get('boundary_notes') or [],
        'accepted_evidence_by_module': selected.get('accepted_evidence_by_module') or {},
        'missing_evidence_by_module': selected.get('missing_evidence_by_module') or {},
        'rejected_evidence_by_module': selected.get('rejected_evidence_by_module') or {},
        'recommended_recipient': recipient,
        'recommended_action': selected.get('recommended_action') or selected['selected_action'],
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(project_owned_boundary.get('third_party_checkpoint_used')),
        'no_sibling_imports': True,
        'label_leaks': leak_terms,
        'label_clean': not leak_terms,
    }
    return build_module_chat_message(
        sender='ai_different',
        recipient=recipient,
        topic=topic,
        body=body,
        evidence={
            'benefit_id': body['benefit_id'],
            'scenario_id': body['scenario_id'],
            'hypothesis_id': body['hypothesis_id'],
            'benefit_classification': body['benefit_classification'],
            'runtime_memory_hash_state': body['runtime_memory_hash_state'],
            'runtime_memory_mutated': body['runtime_memory_mutated'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_campaign_benefit', body['benefit_classification'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_benefit_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_evaluator_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != EVALUATOR_LEDGER_KIND:
        raise ValueError('evaluator ledger has wrong ledger_kind')
    return validate_evaluator_ledger(ledger)


def _valid_prior_outcome_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') not in {EVALUATOR_LEDGER_KIND, 'ai_different.outcome_ledger'}:
        raise ValueError('outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_contract_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CONTRACT_LEDGER_KIND, 'contracts': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CONTRACT_LEDGER_KIND:
        raise ValueError('contract ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_adjudicator_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': ADJUDICATOR_LEDGER_KIND, 'contract_states': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != ADJUDICATOR_LEDGER_KIND:
        raise ValueError('adjudicator ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_agenda_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': AGENDA_LEDGER_KIND, 'hypotheses': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != AGENDA_LEDGER_KIND:
        raise ValueError('agenda ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_lifecycle_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': LIFECYCLE_LEDGER_KIND, 'hypotheses': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != LIFECYCLE_LEDGER_KIND:
        raise ValueError('lifecycle ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_scorecard_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': SCORECARD_LEDGER_KIND, 'scorecards': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != SCORECARD_LEDGER_KIND:
        raise ValueError('scorecard ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CAMPAIGN_LEDGER_KIND, 'campaigns': [], 'campaign_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_LEDGER_KIND:
        raise ValueError('campaign ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_campaign_outcome_or_empty(ledger):
    if not ledger:
        return {'ledger_kind': CAMPAIGN_OUTCOME_LEDGER_KIND, 'outcomes': [], 'outcome_records': [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != CAMPAIGN_OUTCOME_LEDGER_KIND:
        raise ValueError('campaign outcome ledger has wrong ledger_kind')
    return dict(ledger)


def _valid_prior_benefit_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_BENEFIT_LEDGER_KIND:
        raise ValueError('prior benefit ledger has wrong ledger_kind')
    return validate_science_benefit_ledger(ledger)


def _extract_scenarios(existing, messages, campaign_ledger, campaign_outcome_ledger, scorecard_ledger, lifecycle_ledger, agenda_ledger, contract_ledger):
    scenarios: dict[str, dict[str, Any]] = {}
    for item in list(existing or []):
        _upsert_scenario(scenarios, item)
    for item in list(campaign_ledger.get('campaigns') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'campaign_ledger'))
    for item in list(campaign_outcome_ledger.get('outcomes') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'campaign_outcome_ledger'))
    for item in list(scorecard_ledger.get('scorecards') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'scorecard_ledger'))
    for item in list(lifecycle_ledger.get('hypotheses') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'lifecycle_ledger'))
    for item in list(agenda_ledger.get('hypotheses') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'agenda_ledger'))
    for item in list(contract_ledger.get('contracts') or []):
        _upsert_scenario(scenarios, _scenario_from_plain_item(item, 'contract_ledger'))
    for message in messages:
        _upsert_scenario(scenarios, _scenario_from_message(message))
    return [_finalize_scenario(item) for item in scenarios.values()]


def _scenario_from_plain_item(item, source):
    scenario_id = item.get('scenario_id') or item.get('campaign_id') or item.get('hypothesis_id') or item.get('contract_id')
    hypothesis_id = item.get('hypothesis_id')
    contract_id = item.get('contract_id')
    if not hypothesis_id and contract_id:
        hypothesis_id = f'hypothesis:{contract_id}'
    return {
        'scenario_id': str(scenario_id) if scenario_id else None,
        'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
        'isolated': _empty_decision(),
        'connected': _empty_decision(),
        'lineage': [source],
    }


def _scenario_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = message.get('sender')
    scenario_id = body.get('scenario_id') or evidence.get('scenario_id') or body.get('campaign_id') or evidence.get('campaign_id') or body.get('hypothesis_id') or evidence.get('hypothesis_id')
    hypothesis_id = body.get('hypothesis_id') or evidence.get('hypothesis_id')
    if not scenario_id and body.get('contract_id'):
        scenario_id = body.get('contract_id')
    if not hypothesis_id and body.get('contract_id'):
        hypothesis_id = f'hypothesis:{body.get("contract_id")}'
    scenario = {
        'scenario_id': str(scenario_id) if scenario_id else None,
        'hypothesis_id': str(hypothesis_id) if hypothesis_id else None,
        'isolated': _empty_decision(),
        'connected': _empty_decision(),
        'lineage': [f'message:{sender}'],
    }
    decision = _decision_from_message(message)
    if _is_isolated_message(message):
        _merge_decision(scenario['isolated'], decision)
    if _is_connected_message(message):
        _merge_decision(scenario['connected'], decision)
    return scenario


def _upsert_scenario(scenarios, incoming):
    scenario_id = incoming.get('scenario_id')
    if not scenario_id:
        return
    current = scenarios.setdefault(str(scenario_id), {
        'scenario_id': str(scenario_id),
        'hypothesis_id': incoming.get('hypothesis_id'),
        'isolated': _empty_decision(),
        'connected': _empty_decision(),
        'lineage': [],
    })
    current['hypothesis_id'] = current.get('hypothesis_id') or incoming.get('hypothesis_id')
    _merge_decision(current['isolated'], incoming.get('isolated') or _empty_decision())
    _merge_decision(current['connected'], incoming.get('connected') or _empty_decision())
    current['lineage'] = _unique_strings(list(current.get('lineage') or []) + list(incoming.get('lineage') or []))


def _empty_decision():
    return {
        'accepted': [],
        'missing': [],
        'rejected': [],
        'actions': [],
        'evidence_by_module': {},
        'label_leaks': [],
        'third_party_checkpoint_used': False,
    }


def _decision_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    sender = str(message.get('sender'))
    gate = _evidence_gate(message)
    status = _evidence_status(message)
    decision = _empty_decision()
    for field, target in (
        ('accepted_evidence', 'accepted'),
        ('missing_evidence', 'missing'),
        ('rejected_evidence', 'rejected'),
    ):
        values = body.get(field) or evidence.get(field) or []
        if isinstance(values, list):
            decision[target].extend(str(value) for value in values if value)
    if gate in GATE_ORDER:
        if status == 'satisfied':
            decision['accepted'].append(gate)
        elif status == 'missing':
            decision['missing'].append(gate)
        elif status == 'failed':
            decision['rejected'].append(gate)
    action = body.get('selected_action') or body.get('theory_update_action') or evidence.get('selected_action')
    if action:
        decision['actions'].append(str(action))
    leaks = label_leak_terms({'body': body, 'evidence': evidence})
    decision['label_leaks'].extend(leaks)
    decision['third_party_checkpoint_used'] = bool(
        body.get('third_party_checkpoint_used')
        or evidence.get('third_party_checkpoint_used')
    )
    if sender in CONNECTED_SENDERS:
        bucket = decision['evidence_by_module'].setdefault(sender, {'accepted': [], 'missing': [], 'rejected': []})
        if status == 'satisfied':
            bucket['accepted'].append(gate)
        elif status == 'missing':
            bucket['missing'].append(gate)
        elif status == 'failed':
            bucket['rejected'].append(gate)
    return _finalize_decision(decision)


def _is_isolated_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    return message.get('sender') == 'ai_different' and ('isolated' in text or body.get('comparison_mode') == 'isolated')


def _is_connected_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    return message.get('sender') in CONNECTED_SENDERS or 'connected' in text or body.get('comparison_mode') == 'connected'


def _merge_decision(current, incoming):
    for field in ('accepted', 'missing', 'rejected', 'actions', 'label_leaks'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    current['third_party_checkpoint_used'] = bool(current.get('third_party_checkpoint_used') or incoming.get('third_party_checkpoint_used'))
    by_module = dict(current.get('evidence_by_module') or {})
    for module, values in dict(incoming.get('evidence_by_module') or {}).items():
        bucket = by_module.setdefault(module, {'accepted': [], 'missing': [], 'rejected': []})
        for field in ('accepted', 'missing', 'rejected'):
            bucket[field] = _unique_strings(list(bucket.get(field) or []) + list(values.get(field) or []))
    current['evidence_by_module'] = by_module
    _finalize_decision(current)


def _finalize_decision(decision):
    accepted = _unique_strings(decision.get('accepted') or [])
    rejected = _unique_strings(decision.get('rejected') or [])
    observed = bool(
        accepted
        or rejected
        or decision.get('missing')
        or decision.get('actions')
        or decision.get('evidence_by_module')
        or decision.get('label_leaks')
        or decision.get('third_party_checkpoint_used')
    )
    missing = [
        gate for gate in _unique_strings(decision.get('missing') or [])
        if gate in set(GATE_ORDER) and gate not in set(accepted) and gate not in set(rejected)
    ]
    if 'accept_campaign' in set(decision.get('actions') or []):
        accepted = _unique_strings(list(accepted) + list(GATE_ORDER))
    if observed:
        missing = _unique_strings(missing + [gate for gate in GATE_ORDER if gate not in set(accepted) and gate not in set(rejected)])
    decision['accepted'] = accepted
    decision['rejected'] = rejected
    decision['missing'] = missing
    decision['actions'] = _unique_strings(decision.get('actions') or [])
    decision['label_leaks'] = _unique_strings(decision.get('label_leaks') or [])
    return decision


def _finalize_scenario(scenario):
    _finalize_decision(scenario['isolated'])
    _finalize_decision(scenario['connected'])
    scenario['boundary_state'] = _boundary_state(scenario)
    scenario['theory_update_delta'] = _theory_delta(scenario)
    scenario['refinement_delta'] = _delta_for_action(scenario, 'refine_hypothesis')
    scenario['retirement_delta'] = _delta_for_action(scenario, 'retire_theory_line')
    return scenario


def _select_benefit_action(*, scenarios, evaluated_ids, project_owned_boundary):
    for scenario in scenarios:
        if _has_boundary_failure(scenario, project_owned_boundary):
            return _benefit_action(scenario, 'connected_prevents_boundary_or_checkpoint_overclaim', 'repair_boundary', 'code_module')
    for scenario in scenarios:
        if _connected_has_action(scenario, 'retire_theory_line') or (scenario['connected']['rejected'] and not scenario['isolated']['rejected']):
            return _benefit_action(scenario, 'connected_retires_failed_line', 'retire_theory_line', 'broadcast')
    for scenario in scenarios:
        if _all_gates(scenario['connected']) and not _all_gates(scenario['isolated']):
            return _benefit_action(scenario, 'connected_accepts_with_verified_math_code_language', 'accept_campaign', 'broadcast')
    for scenario in scenarios:
        if _connected_has_action(scenario, 'refine_hypothesis') or (
            len(scenario['connected']['accepted']) >= 2
            and scenario['connected']['rejected']
            and not _connected_has_action(scenario, 'retire_theory_line')
        ):
            return _benefit_action(scenario, 'connected_refines_with_clearer_evidence', 'refine_hypothesis', 'broadcast')
    for scenario in scenarios:
        if scenario['connected']['missing'] and scenario.get('scenario_id') not in set(evaluated_ids):
            gate = scenario['connected']['missing'][0]
            return _benefit_action(scenario, 'connected_requests_missing_targeted_evidence', f'request_{gate}', _recipient_for_gate(gate))
    for scenario in scenarios:
        if _all_gates(scenario['isolated']):
            return _benefit_action(scenario, 'isolated_already_sufficient', 'noop', 'broadcast')
    for scenario in scenarios:
        if scenario['connected']['accepted'] or scenario['connected']['missing'] or scenario['connected']['rejected']:
            return _benefit_action(scenario, 'connected_adds_no_safe_benefit', 'noop', 'orchestrator')
    if scenarios:
        return _benefit_action(scenarios[0], 'insufficient_evidence', 'request_more_evidence', 'orchestrator')
    return _noop_action('no scenarios available')


def _benefit_action(scenario, classification, action, recipient):
    connected = scenario['connected']
    isolated = scenario['isolated']
    selected = {
        'benefit_classification': classification,
        'selected_action': action,
        'chosen_recipient': validate_participant(recipient),
        'scenario_id': scenario.get('scenario_id'),
        'hypothesis_id': scenario.get('hypothesis_id'),
        'isolated_decision_summary': _decision_summary(isolated),
        'connected_decision_summary': _decision_summary(connected),
        'sibling_evidence_used': _sibling_evidence_used(connected),
        'accepted_evidence_by_module': _module_field(connected, 'accepted'),
        'missing_evidence_by_module': _module_field(connected, 'missing'),
        'rejected_evidence_by_module': _module_field(connected, 'rejected'),
        'theory_update_delta': scenario.get('theory_update_delta'),
        'refinement_delta': scenario.get('refinement_delta'),
        'retirement_delta': scenario.get('retirement_delta'),
        'missing_evidence': list(connected.get('missing') or []),
        'boundary_notes': _boundary_notes(scenario),
        'boundary_state': scenario.get('boundary_state'),
        'recommended_action': action,
        'label_leaks': _unique_strings(list(isolated.get('label_leaks') or []) + list(connected.get('label_leaks') or [])),
    }
    return selected


def _noop_action(reason):
    return {
        'benefit_classification': 'summarize_noop',
        'selected_action': 'noop',
        'chosen_recipient': None,
        'scenario_id': None,
        'hypothesis_id': None,
        'isolated_decision_summary': {},
        'connected_decision_summary': {},
        'sibling_evidence_used': [],
        'accepted_evidence_by_module': {},
        'missing_evidence_by_module': {},
        'rejected_evidence_by_module': {},
        'theory_update_delta': reason,
        'refinement_delta': 'none',
        'retirement_delta': 'none',
        'missing_evidence': [],
        'boundary_notes': [],
        'boundary_state': 'clean',
        'recommended_action': 'noop',
        'label_leaks': [],
    }


def _has_boundary_failure(scenario, project_owned_boundary):
    if project_owned_boundary.get('third_party_checkpoint_used'):
        return True
    return bool(
        scenario['isolated'].get('label_leaks')
        or scenario['connected'].get('label_leaks')
        or scenario['isolated'].get('third_party_checkpoint_used')
        or scenario['connected'].get('third_party_checkpoint_used')
    )


def _boundary_state(scenario):
    return 'repair' if _has_boundary_failure(scenario, {}) else 'clean'


def _boundary_notes(scenario):
    notes = []
    if scenario['isolated'].get('label_leaks') or scenario['connected'].get('label_leaks'):
        notes.append('label leak detected in comparison evidence')
    if scenario['isolated'].get('third_party_checkpoint_used') or scenario['connected'].get('third_party_checkpoint_used'):
        notes.append('third-party checkpoint boundary must not be overclaimed')
    return notes


def _theory_delta(scenario):
    isolated = _decision_summary(scenario['isolated'])
    connected = _decision_summary(scenario['connected'])
    if isolated == connected:
        return 'no decision delta'
    return f"isolated {isolated.get('state')} -> connected {connected.get('state')}"


def _delta_for_action(scenario, action):
    isolated_has = action in set(scenario['isolated'].get('actions') or [])
    connected_has = action in set(scenario['connected'].get('actions') or [])
    if connected_has and not isolated_has:
        return 'connected_added'
    if isolated_has and connected_has:
        return 'unchanged'
    return 'none'


def _decision_summary(decision):
    if decision.get('label_leaks') or decision.get('third_party_checkpoint_used'):
        state = 'repair'
    elif decision.get('rejected'):
        state = 'rejected'
    elif _all_gates(decision):
        state = 'accepted'
    elif decision.get('missing'):
        state = 'missing'
    else:
        state = 'insufficient'
    return {
        'state': state,
        'accepted': list(decision.get('accepted') or []),
        'missing': list(decision.get('missing') or []),
        'rejected': list(decision.get('rejected') or []),
        'actions': list(decision.get('actions') or []),
    }


def _all_gates(decision):
    return all(gate in set(decision.get('accepted') or []) for gate in GATE_ORDER)


def _connected_has_action(scenario, action):
    return action in set(scenario['connected'].get('actions') or [])


def _recipient_for_gate(gate):
    return {
        'math_proof': 'funfun',
        'code_proof': 'code_module',
        'language_epoch_plan': 'language_model_2',
    }.get(gate, 'orchestrator')


def _module_field(decision, field):
    output = {}
    for module, values in dict(decision.get('evidence_by_module') or {}).items():
        output[module] = list(values.get(field) or [])
    return output


def _sibling_evidence_used(decision):
    used = []
    for module, values in dict(decision.get('evidence_by_module') or {}).items():
        for field in ('accepted', 'missing', 'rejected'):
            for gate in values.get(field) or []:
                used.append({'sender': module, 'evidence_gate': gate, 'status': field})
    return used


def _evidence_gate(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = body.get('evidence_gate') or evidence.get('evidence_gate') or body.get('gate') or evidence.get('gate')
    if explicit:
        return str(explicit)
    sender = message.get('sender')
    if sender == 'funfun':
        return 'math_proof'
    if sender == 'code_module':
        return 'code_proof'
    if sender == 'language_model_2':
        return 'language_epoch_plan'
    return 'advisory'


def _evidence_status(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    explicit = str(body.get('status') or evidence.get('status') or '').lower()
    text = json.dumps({'body': body, 'evidence': evidence, 'tags': message.get('tags')}, sort_keys=True).lower()
    if explicit in {'satisfied', 'resolved', 'passed', 'confirmed', 'accepted'}:
        return 'satisfied'
    if explicit in {'failed', 'blocked', 'contradicted', 'contradiction', 'rejected'}:
        return 'failed'
    if explicit in {'missing', 'needs_clarification', 'clarification_needed'} or 'missing' in text:
        return 'missing'
    if any(token in text for token in ('failed', 'blocked', 'contradict', 'rejected')):
        return 'failed'
    if any(token in text for token in ('satisfied', 'resolved', 'passed', 'confirmed', 'accepted')):
        return 'satisfied'
    return 'advisory'


def _source_hash(evaluator, prior_outcome, contracts, adjudicator, agenda, lifecycle, scorecard, campaign, campaign_outcome, prior_benefit):
    return stable_digest({
        'evaluator': evaluator.get('ledger_hash') or evaluator.get('ledger_id'),
        'prior_outcome': prior_outcome.get('ledger_hash') or prior_outcome.get('ledger_id'),
        'contract': contracts.get('ledger_hash'),
        'adjudicator': adjudicator.get('ledger_hash'),
        'agenda': agenda.get('ledger_hash'),
        'lifecycle': lifecycle.get('ledger_hash'),
        'scorecard': scorecard.get('ledger_hash'),
        'campaign': campaign.get('ledger_hash'),
        'campaign_outcome': campaign_outcome.get('ledger_hash'),
        'prior_benefit': prior_benefit.get('ledger_hash'),
    })


def _find_scenario(scenarios, scenario_id):
    for scenario in scenarios:
        if scenario.get('scenario_id') == scenario_id:
            return scenario
    return None


def _benefit_status_counts(selected, scenarios):
    classification = selected.get('benefit_classification')
    counts = {
        'benefit': 0,
        'no_benefit': 0,
        'insufficient': 0,
        'repair': 0,
    }
    if classification in {
        'connected_retires_failed_line',
        'connected_accepts_with_verified_math_code_language',
        'connected_refines_with_clearer_evidence',
        'connected_requests_missing_targeted_evidence',
    }:
        counts['benefit'] += 1
    elif classification == 'connected_prevents_boundary_or_checkpoint_overclaim':
        counts['repair'] += 1
    elif classification in {'connected_adds_no_safe_benefit', 'isolated_already_sufficient'}:
        counts['no_benefit'] += 1
    elif classification == 'insufficient_evidence':
        counts['insufficient'] += 1
    if classification == 'summarize_noop' and not scenarios:
        counts['insufficient'] += 1
    return counts


def _unique_strings(values):
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
