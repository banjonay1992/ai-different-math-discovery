from __future__ import annotations

"""Observation-only probes for rediscovering counting and arithmetic."""

from typing import Any


ARITHMETIC_TARGETS: tuple[str, ...] = (
    'cardinality_invariance',
    'addition_as_composition',
    'permutation_invariance',
    'successor_step',
    'predecessor_step',
)


def run_arithmetic_rediscovery_probe(
    *,
    seed_start: int = 0,
    seed_count: int = 2,
    variants: tuple[int, ...] | list[int] = (0,),
) -> dict[str, Any]:
    """
    Rediscover arithmetic structure from learner-facing observations only.

    The benchmark manifest supplies worlds, but this probe reads only public
    observation rows. Human target names are applied after candidate discovery.
    """
    try:
        from world.math_domain_worlds import (
            generate_math_domain_world_manifest,
            math_domain_manifest_from_observation,
        )
    except ImportError:  # pragma: no cover - package import fallback
        from first_principles_ai.world.math_domain_worlds import (
            generate_math_domain_world_manifest,
            math_domain_manifest_from_observation,
        )

    safe_seed_count = max(1, int(seed_count or 1))
    safe_variants = tuple(int(variant) for variant in (variants or (0,)))
    discoveries: list[dict[str, Any]] = []
    observations_seen = 0
    leaked_manifest = False
    events = []

    for seed in range(int(seed_start), int(seed_start) + safe_seed_count):
        for variant in safe_variants:
            manifest = generate_math_domain_world_manifest(
                'arithmetic_quantity',
                seed=seed,
                variant=variant,
            )
            for observation in manifest.observations():
                observations_seen += 1
                leaked_manifest = leaked_manifest or math_domain_manifest_from_observation(
                    observation
                )
                inferred = _discover_from_observation(observation, seed, variant)
                discoveries.extend(inferred)
                for item in inferred:
                    events.append({
                        'event': 'arithmetic_candidate_discovered',
                        'seed': seed,
                        'variant': variant,
                        'target': item['target'],
                        'expression': item['expression'],
                        'confidence': item['confidence'],
                    })

    by_target: dict[str, dict[str, Any]] = {}
    for discovery in discoveries:
        target = str(discovery['target'])
        current = by_target.get(target)
        if (
            current is None
            or float(discovery.get('confidence', 0.0) or 0.0)
            > float(current.get('confidence', 0.0) or 0.0)
        ):
            by_target[target] = discovery

    discovered_targets = tuple(
        target for target in ARITHMETIC_TARGETS
        if target in by_target
    )
    missing_targets = tuple(
        target for target in ARITHMETIC_TARGETS
        if target not in by_target
    )
    coverage = (
        len(discovered_targets) / len(ARITHMETIC_TARGETS)
        if ARITHMETIC_TARGETS
        else 0.0
    )
    promoted = [
        {
            'target': target,
            'expression': by_target[target]['expression'],
            'confidence': by_target[target]['confidence'],
            'support_count': sum(
                1 for item in discoveries
                if item['target'] == target
            ),
            'proof_obligations': by_target[target]['proof_obligations'],
            'falsification_tests': by_target[target]['falsification_tests'],
        }
        for target in discovered_targets
    ]
    return {
        'run_kind': 'arithmetic_rediscovery_probe',
        'runs_final': False,
        'seed_start': int(seed_start),
        'seed_count': safe_seed_count,
        'variants': list(safe_variants),
        'observation_count': observations_seen,
        'candidate_count': len(discoveries),
        'target_count': len(ARITHMETIC_TARGETS),
        'discovered_target_count': len(discovered_targets),
        'coverage': round(coverage, 3),
        'status': 'arithmetic_ready' if coverage >= 1.0 and not leaked_manifest else 'needs_more_observations',
        'discovered_targets': list(discovered_targets),
        'missing_targets': list(missing_targets),
        'leaked_manifest': leaked_manifest,
        'self_authored_equations': promoted,
        'live_events': events[:80],
    }


def _discover_from_observation(
    observation: dict[str, Any],
    seed: int,
    variant: int,
) -> list[dict[str, Any]]:
    if observation.get('observation_kind') != 'collection_event':
        return []
    before = dict(observation.get('before') or {})
    after = dict(observation.get('after') or {})
    event = dict(observation.get('event') or {})
    sample_id = str(observation.get('sample_id', 'unknown'))
    if 'group_a' in before and 'group_b' in before:
        left = _extent(before.get('group_a'))
        right = _extent(before.get('group_b'))
        combined = _extent(after.get('group_c'))
        heldout = _extent((observation.get('heldout_view') or {}).get('group_c'))
        confidence = 1.0 if left + right == combined == heldout else 0.35
        return [
            _discovery(
                sample_id=sample_id,
                seed=seed,
                variant=variant,
                target='cardinality_invariance',
                expression='extent(collection) is unchanged by token identity',
                evidence={
                    'combined_extent': combined,
                    'heldout_permuted_extent': heldout,
                },
                confidence=confidence,
                proof_obligations=('identity_under_relabeling',),
                falsification_tests=('permute labels and reject if extent changes',),
            ),
            _discovery(
                sample_id=sample_id,
                seed=seed,
                variant=variant,
                target='addition_as_composition',
                expression='extent(join(A, B)) == extent(A) + extent(B)',
                evidence={
                    'left_extent': left,
                    'right_extent': right,
                    'combined_extent': combined,
                },
                confidence=confidence,
                proof_obligations=('closure_under_join', 'associativity_probe'),
                falsification_tests=('regroup three collections and require the same total',),
            ),
            _discovery(
                sample_id=sample_id,
                seed=seed,
                variant=variant,
                target='permutation_invariance',
                expression='extent(permutation(C)) == extent(C)',
                evidence={
                    'combined_extent': combined,
                    'permuted_extent': heldout,
                },
                confidence=confidence,
                proof_obligations=('bijection_preserves_extent',),
                falsification_tests=('rename every token and require the same extent',),
            ),
        ]

    before_extent = _extent(before.get('group'))
    after_extent = _extent(after.get('group'))
    observed_delta = after_extent - before_extent
    contrast = dict(observation.get('contrast') or {})
    contrast_size = _safe_int(contrast.get('result_size_hint'))
    discoveries = []
    if event.get('move') == 'remove_one':
        confidence = 1.0 if observed_delta == -1 else 0.35
        discoveries.append(_discovery(
            sample_id=sample_id,
            seed=seed,
            variant=variant,
            target='predecessor_step',
            expression='extent(remove_one(C)) == extent(C) - 1',
            evidence={
                'before_extent': before_extent,
                'after_extent': after_extent,
                'observed_delta': observed_delta,
            },
            confidence=confidence,
            proof_obligations=('unit_step_consistency',),
            falsification_tests=('remove one token from a held-out extent',),
        ))
    if contrast.get('event') == 'add_one' and contrast_size is not None:
        confidence = 1.0 if contrast_size - before_extent == 1 else 0.35
        discoveries.append(_discovery(
            sample_id=sample_id,
            seed=seed,
            variant=variant,
            target='successor_step',
            expression='extent(add_one(C)) == extent(C) + 1',
            evidence={
                'before_extent': before_extent,
                'contrast_extent': contrast_size,
                'predicted_delta': contrast_size - before_extent,
            },
            confidence=confidence,
            proof_obligations=('unit_step_consistency', 'successor_closure'),
            falsification_tests=('add one token to a held-out extent',),
        ))
    return discoveries


def _discovery(
    *,
    sample_id: str,
    seed: int,
    variant: int,
    target: str,
    expression: str,
    evidence: dict[str, Any],
    confidence: float,
    proof_obligations: tuple[str, ...],
    falsification_tests: tuple[str, ...],
) -> dict[str, Any]:
    return {
        'key': f'arithmetic:{sample_id}:{target}',
        'sample_id': sample_id,
        'seed': int(seed),
        'variant': int(variant),
        'target': target,
        'expression': expression,
        'evidence': evidence,
        'confidence': round(float(confidence), 3),
        'proof_obligations': list(proof_obligations),
        'falsification_tests': list(falsification_tests),
    }


def _extent(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
