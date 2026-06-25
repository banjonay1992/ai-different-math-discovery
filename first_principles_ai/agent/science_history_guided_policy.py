"""Science interaction-history-guided policy planner.

Turns AI Different-local science coordination interaction-history scorecards
into one cautious next ordering policy. This is not autonomous
self-modification, not a science benchmark, not causal proof, and not a
project-owned checkpoint claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .module_chat_adapter import (
    build_module_chat_message,
    label_leak_terms,
    module_chat_message_id,
    stable_digest,
    validate_module_chat_message,
    validate_participant,
)
from .science_campaign_cycle_strategy_outcome import SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND
from .science_coordination_ab_probe import SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND
from .science_coordination_ab_probe_outcome import SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND
from .science_coordination_history import SCIENCE_COORDINATION_HISTORY_LEDGER_KIND
from .science_coordination_interaction_history import SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND
from .science_coordination_policy_outcome import SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND
from .science_coordination_policy_scorecard import SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND


SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND = 'ai_different.science_history_guided_policy_ledger'


def empty_science_history_guided_policy_ledger() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'ledger_kind': SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND,
        'processed_message_ids': [],
        'processed_source_hashes': [],
        'planned_policy_keys': [],
        'policy_records': [],
        'policy_rows': [],
        'outgoing_response_ids': [],
        'latest': {},
    }


def load_science_history_guided_policy_ledger(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return empty_science_history_guided_policy_ledger()
    with Path(path).open('r', encoding='utf-8') as handle:
        ledger = json.load(handle)
    return validate_science_history_guided_policy_ledger(ledger)


def write_science_history_guided_policy_ledger(path: str | Path, ledger: dict[str, Any]) -> Path:
    validated = validate_science_history_guided_policy_ledger(ledger)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def validate_science_history_guided_policy_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError('science history-guided policy ledger must be a JSON object')
    if ledger.get('ledger_kind') != SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND:
        raise ValueError('science history-guided policy ledger has wrong ledger_kind')
    for field in ('processed_message_ids', 'processed_source_hashes', 'planned_policy_keys', 'outgoing_response_ids'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f'{field} must be strings')
    for field in ('policy_records', 'policy_rows'):
        values = ledger.get(field)
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError(f'{field} must be objects')
    latest = ledger.get('latest', {})
    if not isinstance(latest, dict):
        raise ValueError('science history-guided policy latest must be an object')
    validated = {
        'schema_version': int(ledger.get('schema_version', 1) or 1),
        'ledger_kind': SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND,
        'processed_message_ids': _unique_strings(ledger['processed_message_ids']),
        'processed_source_hashes': _unique_strings(ledger['processed_source_hashes']),
        'planned_policy_keys': _unique_strings(ledger['planned_policy_keys']),
        'policy_records': list(ledger['policy_records']),
        'policy_rows': list(ledger['policy_rows']),
        'outgoing_response_ids': _unique_strings(ledger['outgoing_response_ids']),
        'latest': dict(latest),
    }
    if ledger.get('ledger_hash'):
        validated['ledger_hash'] = str(ledger['ledger_hash'])
    if ledger.get('artifact_path'):
        validated['artifact_path'] = str(ledger['artifact_path'])
    return validated


def read_science_history_guided_policy_transcript(path: str | Path | None) -> dict[str, Any]:
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
        raise ValueError('plain science history-guided policy input must be a JSON object')
    return value


def build_science_history_guided_policy_plan(
    *,
    transcript_messages: list[dict[str, Any]],
    history_guided_policy_ledger: dict[str, Any],
    interaction_history_ledger: dict[str, Any] | None = None,
    ab_probe_outcome_ledger: dict[str, Any] | None = None,
    ab_probe_ledger: dict[str, Any] | None = None,
    policy_scorecard_ledger: dict[str, Any] | None = None,
    policy_outcome_ledger: dict[str, Any] | None = None,
    history_ledger: dict[str, Any] | None = None,
    cycle_strategy_outcome_ledger: dict[str, Any] | None = None,
    theory_memory_ledger: dict[str, Any] | None = None,
    campaign_ledger: dict[str, Any] | None = None,
    hypothesis_ledger: dict[str, Any] | None = None,
    experiment_ledger: dict[str, Any] | None = None,
    module_chat_ledger: dict[str, Any] | None = None,
    prior_history_guided_policy_ledger: dict[str, Any] | None = None,
    runtime_memory_data: dict[str, Any] | None = None,
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ledger = validate_science_history_guided_policy_ledger(history_guided_policy_ledger)
    interaction_history = _valid_kind_or_empty(interaction_history_ledger or {}, SCIENCE_COORDINATION_INTERACTION_HISTORY_LEDGER_KIND, 'scorecard_records')
    ab_probe_outcome = _valid_kind_or_empty(ab_probe_outcome_ledger or {}, SCIENCE_COORDINATION_AB_PROBE_OUTCOME_LEDGER_KIND, 'outcome_records')
    ab_probe = _valid_kind_or_empty(ab_probe_ledger or {}, SCIENCE_COORDINATION_AB_PROBE_LEDGER_KIND, 'probe_records')
    policy_scorecard = _valid_kind_or_empty(policy_scorecard_ledger or {}, SCIENCE_COORDINATION_POLICY_SCORECARD_LEDGER_KIND, 'scorecard_records')
    policy_outcome = _valid_kind_or_empty(policy_outcome_ledger or {}, SCIENCE_COORDINATION_POLICY_OUTCOME_LEDGER_KIND, 'outcome_records')
    history = _valid_kind_or_empty(history_ledger or {}, SCIENCE_COORDINATION_HISTORY_LEDGER_KIND, 'event_records')
    cycle_outcome = _valid_kind_or_empty(cycle_strategy_outcome_ledger or {}, SCIENCE_CAMPAIGN_CYCLE_STRATEGY_OUTCOME_LEDGER_KIND, 'outcome_records')
    theory_memory = _valid_plain_or_empty(theory_memory_ledger or {})
    campaign = _valid_plain_or_empty(campaign_ledger or {})
    hypothesis = _valid_plain_or_empty(hypothesis_ledger or {})
    experiment = _valid_plain_or_empty(experiment_ledger or {})
    module_chat = _valid_plain_or_empty(module_chat_ledger or {})
    prior_policy = _valid_prior_policy_or_empty(prior_history_guided_policy_ledger or {})
    runtime_memory = dict(runtime_memory_data or {})
    hf_status = dict(hf_use_status or {'hf_validation_used': False})
    source_hash = _source_hash(
        interaction_history,
        ab_probe_outcome,
        ab_probe,
        policy_scorecard,
        policy_outcome,
        history,
        cycle_outcome,
        theory_memory,
        campaign,
        hypothesis,
        experiment,
        module_chat,
        prior_policy,
        runtime_memory,
        hf_status,
    )
    source_is_new = source_hash not in set(ledger['processed_source_hashes'])
    processed = set(ledger['processed_message_ids'])
    new_messages = [message for message in transcript_messages if module_chat_message_id(message) not in processed]
    skipped_messages = [message for message in transcript_messages if module_chat_message_id(message) in processed]
    rows = _extract_policy_rows(
        ledger['policy_rows'],
        interaction_history,
        ab_probe_outcome,
        ab_probe,
        policy_scorecard,
        policy_outcome,
        history,
        cycle_outcome,
        theory_memory,
        campaign,
        hypothesis,
        experiment,
        module_chat,
        transcript_messages,
    )
    if not new_messages and not source_is_new:
        selected = _noop_policy('no new science history-guided policy evidence or source ledger state')
    else:
        selected = _select_policy(rows=rows, project_owned_boundary=project_owned_boundary)
    policy_id = 'science_history_guided_policy_' + stable_digest({
        'source_hash': source_hash,
        'new_message_ids': [module_chat_message_id(message) for message in new_messages],
        'decision': selected['selected_policy_decision'],
        'ordering': selected.get('selected_ordering_key'),
    })[:16]
    selected['science_history_guided_policy_id'] = policy_id
    message = export_science_history_guided_policy_message(
        selected,
        rows=rows,
        runtime_memory_hash_state=runtime_memory_hash_state,
        project_owned_boundary=project_owned_boundary,
        hf_use_status=hf_status,
        source_hashes={
            'science_history_guided_policy_source_hash': source_hash,
            'interaction_history_ledger_hash': interaction_history.get('ledger_hash'),
            'ab_probe_outcome_ledger_hash': ab_probe_outcome.get('ledger_hash'),
            'ab_probe_ledger_hash': ab_probe.get('ledger_hash'),
            'policy_scorecard_ledger_hash': policy_scorecard.get('ledger_hash'),
            'policy_outcome_ledger_hash': policy_outcome.get('ledger_hash'),
            'history_ledger_hash': history.get('ledger_hash'),
            'cycle_strategy_outcome_ledger_hash': cycle_outcome.get('ledger_hash'),
            'theory_memory_ledger_hash': theory_memory.get('ledger_hash'),
            'campaign_ledger_hash': campaign.get('ledger_hash'),
            'hypothesis_ledger_hash': hypothesis.get('ledger_hash'),
            'experiment_ledger_hash': experiment.get('ledger_hash'),
            'module_chat_ledger_hash': module_chat.get('ledger_hash'),
            'prior_history_guided_policy_ledger_hash': prior_policy.get('ledger_hash'),
        },
    )
    new_ids = [module_chat_message_id(message) for message in new_messages]
    outgoing_id = module_chat_message_id(message) if message else None
    policy_key = _policy_key(selected)
    if selected['selected_policy_decision'] != 'summarize_noop':
        ledger['planned_policy_keys'] = _unique_strings(list(ledger['planned_policy_keys']) + [policy_key])
    if new_messages:
        ledger['processed_message_ids'] = _unique_strings(list(ledger['processed_message_ids']) + new_ids)
    if source_is_new:
        ledger['processed_source_hashes'] = _unique_strings(list(ledger['processed_source_hashes']) + [source_hash])
    if outgoing_id:
        ledger['outgoing_response_ids'] = _unique_strings(list(ledger['outgoing_response_ids']) + [outgoing_id])
    state_counts = _state_counts(selected)
    latest = {
        'science_history_guided_policy_id': policy_id,
        'new_message_count': len(new_messages),
        'skipped_message_count': len(skipped_messages),
        'source_is_new': source_is_new,
        'state_counts': state_counts,
        'selected_policy_decision': selected['selected_policy_decision'],
        'selected_action': selected['selected_action'],
        'chosen_recipient': message.get('recipient') if message else None,
        'outbox_count': 1 if message else 0,
        'label_leaks': selected.get('label_leaks', []),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'hf_use_status': hf_status,
    }
    record = {
        'science_history_guided_policy_id': policy_id,
        'policy_hash': stable_digest({'policy_id': policy_id, 'selected': selected}),
        'processed_message_ids': new_ids,
        'source_hash': source_hash,
        'source_interaction_history_ids': selected.get('source_interaction_history_ids'),
        'selected_ordering_key': selected.get('selected_ordering_key'),
        'selected_policy_decision': selected['selected_policy_decision'],
        'selected_ordering': selected.get('selected_ordering'),
        'retired_ordering': selected.get('retired_ordering'),
        'target_campaign_ids': selected.get('target_campaign_ids'),
        'target_hypothesis_ids': selected.get('target_hypothesis_ids'),
        'target_simulation_ids': selected.get('target_simulation_ids'),
        'required_evidence_gate': selected.get('required_evidence_gate'),
        'stop_condition': selected.get('stop_condition'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'candidate_not_causal_wording': selected.get('candidate_not_causal_wording'),
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'hf_use_status': hf_status,
        'outgoing_response_id': outgoing_id,
    }
    ledger['policy_rows'] = rows
    if new_messages or source_is_new or message is not None:
        ledger['policy_records'] = list(ledger['policy_records']) + [record]
    ledger['latest'] = latest
    if artifact_path is not None:
        ledger['artifact_path'] = str(Path(artifact_path))
    ledger['ledger_hash'] = stable_digest({key: value for key, value in ledger.items() if key != 'ledger_hash'})
    return ledger, message


def export_science_history_guided_policy_message(
    selected: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    runtime_memory_hash_state: dict[str, Any],
    project_owned_boundary: dict[str, Any],
    hf_use_status: dict[str, Any],
    source_hashes: dict[str, Any],
    topic: str = 'ai_different.science_history_guided_policy',
) -> dict[str, Any] | None:
    if selected['selected_policy_decision'] == 'summarize_noop':
        return None
    recipient = validate_participant(selected.get('selected_recipient') or 'orchestrator')
    row = _find_row(rows, selected.get('selected_ordering_key')) or {}
    leak_terms = label_leak_terms({'selected': selected, 'row': row})
    body = {
        'module': 'AI Different',
        'response_kind': 'science_history_guided_policy',
        'science_history_guided_policy_id': selected.get('science_history_guided_policy_id'),
        'selected_policy_decision': selected['selected_policy_decision'],
        'selected_action': selected['selected_action'],
        'selected_recipient': recipient,
        'selected_ordering_key': selected.get('selected_ordering_key'),
        'selected_ordering': selected.get('selected_ordering'),
        'retired_ordering': selected.get('retired_ordering'),
        'source_interaction_history_ids': selected.get('source_interaction_history_ids') or [],
        'target_campaign_ids': selected.get('target_campaign_ids') or [],
        'target_hypothesis_ids': selected.get('target_hypothesis_ids') or [],
        'target_simulation_ids': selected.get('target_simulation_ids') or [],
        'required_evidence_gate': selected.get('required_evidence_gate'),
        'stop_condition': selected.get('stop_condition'),
        'tested_commits': selected.get('tested_commits') or [],
        'tested_tests': selected.get('tested_tests') or [],
        'candidate_not_causal': True,
        'candidate_not_causal_wording': selected.get('candidate_not_causal_wording'),
        'checkpoint_boundary_state': selected.get('checkpoint_boundary_state'),
        'checkpoint_boundary_notes': selected.get('checkpoint_boundary_notes') or [],
        'no_overclaiming_proof': selected.get('no_overclaiming_proof'),
        'source_ledger_hashes': dict(source_hashes),
        'runtime_memory_hash_state': dict(runtime_memory_hash_state),
        'runtime_memory_mutated': not bool(runtime_memory_hash_state.get('unchanged', True)),
        'project_owned_boundary': dict(project_owned_boundary),
        'third_party_checkpoint_used': bool(project_owned_boundary.get('third_party_checkpoint_used')),
        'project_owned_checkpoint_claimed': bool(project_owned_boundary.get('project_owned_checkpoint_verified')),
        'hf_use_status': dict(hf_use_status),
        'hf_validation_used': bool(hf_use_status.get('hf_validation_used')),
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
            'science_history_guided_policy_id': body['science_history_guided_policy_id'],
            'selected_policy_decision': body['selected_policy_decision'],
            'selected_ordering_key': body['selected_ordering_key'],
            'candidate_not_causal': True,
            'hf_validation_used': body['hf_validation_used'],
            'label_clean': body['label_clean'],
            'no_sibling_imports': True,
        },
        tags=['ai_different', 'science_history_guided_policy', body['selected_policy_decision'], 'label_clean' if body['label_clean'] else 'label_review_needed'],
    )


def write_science_history_guided_policy_outbox_jsonl(path: str | Path, message: dict[str, Any] | None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((json.dumps(message, sort_keys=True) + '\n') if message is not None else '', encoding='utf-8')
    return output


def _valid_kind_or_empty(ledger, kind, record_field):
    if not ledger:
        return {'ledger_kind': kind, record_field: [], 'ledger_hash': None}
    if ledger.get('ledger_kind') != kind:
        raise ValueError(f'plain science history-guided policy source has wrong ledger_kind: {ledger.get("ledger_kind")}')
    return dict(ledger)


def _valid_plain_or_empty(ledger):
    if not ledger:
        return {}
    if not isinstance(ledger, dict):
        raise ValueError('plain science history-guided policy source must be a JSON object')
    return dict(ledger)


def _valid_prior_policy_or_empty(ledger):
    if not ledger:
        return {}
    if ledger.get('ledger_kind') != SCIENCE_HISTORY_GUIDED_POLICY_LEDGER_KIND:
        raise ValueError('prior science history-guided policy ledger has wrong ledger_kind')
    return validate_science_history_guided_policy_ledger(ledger)


def _extract_policy_rows(existing, interaction_history, ab_probe_outcome, ab_probe, policy_scorecard, policy_outcome, history, cycle_outcome, theory_memory, campaign, hypothesis, experiment, module_chat, messages):
    rows: dict[str, dict[str, Any]] = {}
    for row in list(existing or []):
        _upsert_row(rows, row)
    for record in list(interaction_history.get('scorecard_records') or []):
        _upsert_row(rows, _row_from_source(record, 'interaction_history'))
    for record in list(interaction_history.get('ordering_rows') or []):
        _upsert_row(rows, _row_from_source(record, 'interaction_history_row'))
    for record in list(ab_probe_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_source(record, 'ab_probe_outcome'))
    for record in list(ab_probe.get('probe_records') or []):
        _upsert_row(rows, _row_from_source(record, 'ab_probe'))
    for record in list(policy_scorecard.get('scorecard_records') or []):
        _upsert_row(rows, _row_from_source(record, 'policy_scorecard'))
    for record in list(policy_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_source(record, 'policy_outcome'))
    for record in list(history.get('event_records') or []):
        _upsert_row(rows, _row_from_source(record, 'history'))
    for record in list(cycle_outcome.get('outcome_records') or []):
        _upsert_row(rows, _row_from_source(record, 'cycle_strategy_outcome'))
    for row in _plain_rows(theory_memory, ('theory_memory_rows', 'theory_records', 'theories')):
        _upsert_row(rows, _row_from_source(row, 'theory_memory'))
    for row in _plain_rows(campaign, ('campaign_rows', 'campaigns', 'campaign_records')):
        _upsert_row(rows, _row_from_source(row, 'campaign'))
    for row in _plain_rows(hypothesis, ('hypothesis_rows', 'hypotheses', 'hypothesis_records')):
        _upsert_row(rows, _row_from_source(row, 'hypothesis'))
    for row in _plain_rows(experiment, ('experiment_rows', 'simulation_rows', 'simulation_results', 'requests')):
        _upsert_row(rows, _row_from_source(row, 'experiment'))
    for row in _plain_rows(module_chat, ('messages', 'records', 'module_chat_rows')):
        if isinstance(row, dict) and {'sender', 'recipient', 'topic', 'body', 'evidence'}.issubset(row):
            _upsert_row(rows, _row_from_message(row))
    for message in messages:
        _upsert_row(rows, _row_from_message(message))
    return [_finalize_row(row) for row in rows.values()]


def _row_from_message(message):
    body = dict(message.get('body') or {})
    evidence = dict(message.get('evidence') or {})
    source = {**body, **evidence}
    source['source_interaction_history_id'] = body.get('science_coordination_interaction_history_id') or evidence.get('science_coordination_interaction_history_id')
    source['lineage'] = [f"message:{message.get('sender')}"]
    source['label_leaks'] = label_leak_terms({'body': body, 'evidence': evidence})
    return _row_from_source(source, 'message')


def _row_from_source(source, lineage):
    source = dict(source or {})
    ordering_key = _ordering_key_from_source(source)
    selected = source.get('selected_decision') or source.get('selected_action') or source.get('retention_decision') or source.get('status')
    retention = source.get('retention_decision')
    signal = _signal_from_value(selected, retention)
    if bool(source.get('third_party_checkpoint_used')) or source.get('checkpoint_boundary_state') == 'repair':
        signal = 'boundary'
    target_campaign_ids = _ids_from_source(source, 'campaign')
    target_hypothesis_ids = _ids_from_source(source, 'hypothesis')
    target_simulation_ids = _ids_from_source(source, 'simulation')
    return {
        'selected_ordering_key': ordering_key,
        'source_interaction_history_ids': _unique_strings(list(source.get('source_interaction_history_ids') or []) + ([source.get('source_interaction_history_id')] if source.get('source_interaction_history_id') else []) + ([source.get('science_coordination_interaction_history_id')] if source.get('science_coordination_interaction_history_id') else [])),
        'target_campaign_ids': target_campaign_ids,
        'target_hypothesis_ids': target_hypothesis_ids,
        'target_simulation_ids': target_simulation_ids,
        'signal_counts': {'retained': 1 if signal == 'retained' else 0, 'weakened': 1 if signal == 'weakened' else 0, 'retired': 1 if signal == 'retired' else 0, 'waiting': 1 if signal == 'waiting' else 0, 'no_gain': 1 if signal == 'no_gain' else 0, 'boundary': 1 if signal == 'boundary' else 0},
        'tested_commits': list(source.get('tested_commits') or source.get('source_commits') or []),
        'tested_tests': list(source.get('tested_tests') or source.get('source_tests') or []),
        'checkpoint_boundary_state': 'repair' if signal == 'boundary' else source.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(source.get('checkpoint_boundary_notes') or (['checkpoint boundary preserved'] if signal == 'boundary' else [])),
        'label_leaks': list(source.get('label_leaks') or []),
        'lineage': list(source.get('lineage') or [lineage]),
    }


def _upsert_row(rows, incoming):
    key = incoming.get('selected_ordering_key') or stable_digest(incoming)[:12]
    current = rows.setdefault(str(key), {
        'selected_ordering_key': incoming.get('selected_ordering_key'),
        'source_interaction_history_ids': [],
        'target_campaign_ids': [],
        'target_hypothesis_ids': [],
        'target_simulation_ids': [],
        'signal_counts': {'retained': 0, 'weakened': 0, 'retired': 0, 'waiting': 0, 'no_gain': 0, 'boundary': 0},
        'tested_commits': [],
        'tested_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'label_leaks': [],
        'lineage': [],
    })
    for field in ('source_interaction_history_ids', 'target_campaign_ids', 'target_hypothesis_ids', 'target_simulation_ids', 'tested_commits', 'tested_tests', 'checkpoint_boundary_notes', 'label_leaks', 'lineage'):
        current[field] = _unique_strings(list(current.get(field) or []) + list(incoming.get(field) or []))
    counts = dict(current.get('signal_counts') or {})
    for key, value in dict(incoming.get('signal_counts') or {}).items():
        counts[key] = int(counts.get(key, 0) or 0) + int(value or 0)
    current['signal_counts'] = counts
    current['checkpoint_boundary_state'] = 'repair' if current.get('checkpoint_boundary_state') == 'repair' or incoming.get('checkpoint_boundary_state') == 'repair' else 'clean'


def _finalize_row(row):
    if row.get('label_leaks'):
        row['checkpoint_boundary_state'] = 'repair'
        row['signal_counts']['boundary'] = int(row['signal_counts'].get('boundary', 0) or 0) + 1
    row['history_guided_policy_row_hash'] = stable_digest({
        'ordering': row.get('selected_ordering_key'),
        'signals': row.get('signal_counts'),
        'boundary': row.get('checkpoint_boundary_state'),
    })
    return row


def _select_policy(*, rows, project_owned_boundary):
    for row in rows:
        if project_owned_boundary.get('third_party_checkpoint_used') or row.get('checkpoint_boundary_state') == 'repair' or _count(row, 'boundary'):
            boundary_row = dict(row)
            boundary_row['checkpoint_boundary_state'] = 'repair'
            boundary_row['checkpoint_boundary_notes'] = _unique_strings(list(boundary_row.get('checkpoint_boundary_notes') or []) + ['checkpoint boundary preserved'])
            return _policy(boundary_row, 'preserve_checkpoint_boundary')
    for row in rows:
        if row.get('selected_ordering_key') == 'code_simulation_before_science' and _count(row, 'retained'):
            return _policy(row, 'apply_retained_code_simulation_before_science_ordering')
    for row in rows:
        if row.get('selected_ordering_key') == 'proof_before_hypothesis' and _count(row, 'retained'):
            return _policy(row, 'apply_retained_proof_before_hypothesis_ordering')
    for row in rows:
        if row.get('selected_ordering_key') == 'language_summary_before_science' and _count(row, 'retained'):
            return _policy(row, 'apply_retained_language_summary_before_science_ordering')
    for row in rows:
        if row.get('selected_ordering_key') == 'history_policy_memory_before_science' and _count(row, 'retained'):
            return _policy(row, 'apply_retained_history_policy_memory_ordering')
    for row in rows:
        if _count(row, 'retired') or _count(row, 'weakened'):
            return _policy(row, 'weaken_or_retire_ordering_after_control_wins')
    for row in rows:
        if _count(row, 'waiting'):
            return _policy(row, 'keep_history_guided_policy_waiting_for_more_evidence')
    for row in rows:
        if _count(row, 'no_gain'):
            return _policy(row, 'record_no_actionable_history_policy_gain')
    return _noop_policy('no science history-guided policy selected')


def _policy(row, selected_decision):
    ordering = row.get('selected_ordering_key')
    retired = ordering if selected_decision == 'weaken_or_retire_ordering_after_control_wins' else None
    selected_ordering = None if retired else ordering
    return {
        'selected_policy_decision': selected_decision,
        'selected_action': selected_decision,
        'selected_recipient': 'orchestrator',
        'selected_ordering_key': ordering,
        'selected_ordering': selected_ordering,
        'retired_ordering': retired,
        'source_interaction_history_ids': list(row.get('source_interaction_history_ids') or []),
        'target_campaign_ids': list(row.get('target_campaign_ids') or []),
        'target_hypothesis_ids': list(row.get('target_hypothesis_ids') or []),
        'target_simulation_ids': list(row.get('target_simulation_ids') or []),
        'required_evidence_gate': _evidence_gate(selected_decision),
        'stop_condition': _stop_condition(selected_decision),
        'tested_commits': list(row.get('tested_commits') or []),
        'tested_tests': list(row.get('tested_tests') or []),
        'checkpoint_boundary_state': row.get('checkpoint_boundary_state') or 'clean',
        'checkpoint_boundary_notes': list(row.get('checkpoint_boundary_notes') or []),
        'candidate_not_causal_wording': 'Candidate-not-causal history-guided policy: local interaction history can guide the next ordering to try, not prove causality.',
        'no_overclaiming_proof': 'candidate-not-causal science history-guided policy only; no autonomous self-modification, no science benchmark claim, and no project-owned checkpoint claim without local verification',
        'label_leaks': list(row.get('label_leaks') or []),
    }


def _noop_policy(reason):
    return {
        'selected_policy_decision': 'summarize_noop',
        'selected_action': 'summarize_noop',
        'selected_recipient': None,
        'selected_ordering_key': None,
        'selected_ordering': None,
        'retired_ordering': None,
        'source_interaction_history_ids': [],
        'target_campaign_ids': [],
        'target_hypothesis_ids': [],
        'target_simulation_ids': [],
        'required_evidence_gate': None,
        'stop_condition': reason,
        'tested_commits': [],
        'tested_tests': [],
        'checkpoint_boundary_state': 'clean',
        'checkpoint_boundary_notes': [],
        'candidate_not_causal_wording': reason,
        'no_overclaiming_proof': reason,
        'label_leaks': [],
    }


def _state_counts(selected):
    counts = {'applied': 0, 'weakened': 0, 'retired': 0, 'waiting': 0, 'boundary': 0, 'no_gain': 0}
    decision = selected.get('selected_policy_decision')
    if decision == 'preserve_checkpoint_boundary':
        counts['boundary'] += 1
    elif decision in {
        'apply_retained_code_simulation_before_science_ordering',
        'apply_retained_proof_before_hypothesis_ordering',
        'apply_retained_language_summary_before_science_ordering',
        'apply_retained_history_policy_memory_ordering',
    }:
        counts['applied'] += 1
    elif decision == 'weaken_or_retire_ordering_after_control_wins':
        counts['weakened'] += 1
        counts['retired'] += 1
    elif decision == 'keep_history_guided_policy_waiting_for_more_evidence':
        counts['waiting'] += 1
    elif decision in {'record_no_actionable_history_policy_gain', 'summarize_noop'}:
        counts['no_gain'] += 1
    return counts


def _ordering_key_from_source(source):
    explicit = source.get('selected_ordering_key') or source.get('interaction_ordering_key') or source.get('ordering_key')
    if explicit:
        return str(explicit)
    selected = source.get('selected_decision') or source.get('selected_policy_decision') or source.get('selected_action') or source.get('policy_class') or ''
    ordering = str(source.get('selected_ordering') or source.get('retained_ordering') or source.get('retired_ordering') or '')
    text = f'{selected} {ordering}'.lower()
    if 'code' in text or 'simulation' in text:
        return 'code_simulation_before_science'
    if 'proof' in text or 'formal' in text or 'funfun' in text:
        return 'proof_before_hypothesis'
    if 'language' in text or 'terminology' in text or 'summary' in text:
        return 'language_summary_before_science'
    if 'history' in text or 'memory' in text:
        return 'history_policy_memory_before_science'
    return 'unknown_ordering'


def _signal_from_value(selected, retention):
    mapping = {
        'retain_code_simulation_before_science_when_repeatedly_helpful': 'retained',
        'retain_proof_before_hypothesis_when_repeatedly_helpful': 'retained',
        'retain_language_summary_before_science_when_repeatedly_helpful': 'retained',
        'retain_history_policy_memory_when_repeatedly_helpful': 'retained',
        'retained': 'retained',
        'weaken_or_retire_ordering_after_repeated_control_wins': 'retired',
        'weakened': 'weakened',
        'retired': 'retired',
        'keep_ordering_underpowered_waiting_for_more_evidence': 'waiting',
        'waiting': 'waiting',
        'record_no_measurable_interaction_gain': 'no_gain',
        'no_gain': 'no_gain',
        'preserve_checkpoint_boundary': 'boundary',
    }
    if selected in mapping:
        return mapping[selected]
    if retention in mapping:
        return mapping[retention]
    return None


def _evidence_gate(decision):
    mapping = {
        'preserve_checkpoint_boundary': 'verify_no_project_owned_checkpoint_overclaim',
        'apply_retained_code_simulation_before_science_ordering': 'code_simulation_result_before_science_campaign',
        'apply_retained_proof_before_hypothesis_ordering': 'funfun_formal_or_proof_certificate_before_hypothesis',
        'apply_retained_language_summary_before_science_ordering': 'language_protocol_summary_before_science_campaign',
        'apply_retained_history_policy_memory_ordering': 'history_policy_memory_receipt_before_campaign',
        'weaken_or_retire_ordering_after_control_wins': 'confirm_retired_ordering_not_reused_without_new_evidence',
        'keep_history_guided_policy_waiting_for_more_evidence': 'collect_more_interaction_history_evidence',
        'record_no_actionable_history_policy_gain': 'record_no_actionable_gain',
    }
    return mapping.get(decision)


def _stop_condition(decision):
    if decision == 'preserve_checkpoint_boundary':
        return 'stop when checkpoint boundary is clean and no project-owned model is claimed'
    if decision.startswith('apply_retained_'):
        return 'stop after one bounded campaign cycle or when required evidence gate is satisfied'
    if decision == 'weaken_or_retire_ordering_after_control_wins':
        return 'stop when ordering is marked weakened or retired for the next cycle'
    if decision == 'keep_history_guided_policy_waiting_for_more_evidence':
        return 'stop after collecting one more relevant interaction-history outcome'
    return 'stop after recording no actionable history-policy gain'


def _ids_from_source(source, prefix):
    plural = source.get(f'target_{prefix}_ids') or source.get(f'{prefix}_ids')
    if isinstance(plural, list):
        return _unique_strings(plural)
    single = source.get(f'{prefix}_id') or source.get(f'target_{prefix}_id')
    return [str(single)] if single else []


def _count(row, key):
    return int((row.get('signal_counts') or {}).get(key, 0) or 0)


def _source_hash(*sources):
    def digest_or_hash(value):
        if not value:
            return None
        if isinstance(value, dict):
            return value.get('ledger_hash') or stable_digest(value)
        return stable_digest(value)

    return stable_digest([digest_or_hash(source) for source in sources])


def _plain_rows(source, keys):
    rows = []
    for key in keys:
        values = source.get(key)
        if isinstance(values, list):
            rows.extend([dict(item) for item in values if isinstance(item, dict)])
    return rows


def _find_row(rows, ordering_key):
    for row in rows:
        if ordering_key and row.get('selected_ordering_key') == ordering_key:
            return row
    return None


def _policy_key(selected):
    return stable_digest({
        'ordering': selected.get('selected_ordering_key'),
        'decision': selected.get('selected_policy_decision'),
    })


def _unique_strings(values):
    seen = set()
    output = []
    for value in values:
        text = str(value)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return output
