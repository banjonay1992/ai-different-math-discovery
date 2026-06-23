from __future__ import annotations

"""
Cross-experiment memory for learned dynamics laws.

The memory is deliberately explicit:
- episodic memory stores what happened in a particular run
- semantic memory consolidates repeated law families into reusable schemas
- transfer reports compare a new run against what memory would have expected
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentRecord:
    """One completed world run stored as episodic memory."""
    episode_id: str
    world_type: str
    seed: int
    object_count: int
    step_count: int
    learned_laws: list[dict]
    confirmed_novel_types: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    transfer_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'episode_id': self.episode_id,
            'world_type': self.world_type,
            'seed': self.seed,
            'object_count': self.object_count,
            'step_count': self.step_count,
            'learned_laws': self.learned_laws,
            'confirmed_novel_types': list(self.confirmed_novel_types),
            'diagnostics': self.diagnostics,
            'transfer_report': self.transfer_report,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentRecord':
        return cls(
            episode_id=str(data.get('episode_id', 'E_000')),
            world_type=str(data.get('world_type', 'unknown')),
            seed=int(data.get('seed', 0)),
            object_count=int(data.get('object_count', 0)),
            step_count=int(data.get('step_count', 0)),
            learned_laws=list(data.get('learned_laws', [])),
            confirmed_novel_types=tuple(data.get('confirmed_novel_types', ())),
            diagnostics=dict(data.get('diagnostics', {})),
            transfer_report=dict(data.get('transfer_report', {})),
        )


@dataclass
class LawSchema:
    """A consolidated semantic memory for one family of dynamics laws."""
    law_type: str
    support_count: int = 0
    source_episodes: list[str] = field(default_factory=list)
    source_worlds: set[str] = field(default_factory=set)
    absent_worlds: set[str] = field(default_factory=set)
    best_confidence: float = 0.0
    total_improvement: float = 0.0
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    categorical_values: dict[str, set[str]] = field(default_factory=dict)
    component_types: set[str] = field(default_factory=set)

    @property
    def mean_improvement(self) -> float:
        if self.support_count == 0:
            return 0.0
        return self.total_improvement / self.support_count

    @property
    def transfer_score(self) -> float:
        total_contexts = len(self.source_worlds) + len(self.absent_worlds)
        if total_contexts == 0:
            coverage = 0.0
        else:
            coverage = len(self.source_worlds) / total_contexts
        strength = 0.65 * self.best_confidence + 0.35 * self.mean_improvement
        return max(0.0, min(1.0, strength * (0.7 + 0.3 * coverage)))

    def update_from_law(self, episode_id: str, world_type: str, law: dict):
        self.support_count += 1
        self.source_episodes.append(episode_id)
        self.source_worlds.add(world_type)
        self.absent_worlds.discard(world_type)
        self.best_confidence = max(self.best_confidence, float(law.get('confidence', 0.0)))
        self.total_improvement += float(law.get('improvement', 0.0))

        properties = law.get('properties', {})
        for key in ('center_x', 'center_y', 'period_steps', 'amplitude', 'component_count'):
            if key in properties and isinstance(properties[key], (int, float)):
                self._extend_range(key, float(properties[key]))
        for key in ('direction', 'spin', 'axis', 'term_origin'):
            if key in properties:
                self.categorical_values.setdefault(key, set()).add(str(properties[key]))
        for component in properties.get('components', []):
            self.component_types.add(str(component))

    def _extend_range(self, key: str, value: float):
        if key not in self.parameter_ranges:
            self.parameter_ranges[key] = (value, value)
            return
        low, high = self.parameter_ranges[key]
        self.parameter_ranges[key] = (min(low, value), max(high, value))

    def to_dict(self) -> dict:
        return {
            'law_type': self.law_type,
            'support_count': self.support_count,
            'source_worlds': sorted(self.source_worlds),
            'absent_worlds': sorted(self.absent_worlds),
            'best_confidence': round(self.best_confidence, 3),
            'total_improvement': round(self.total_improvement, 6),
            'mean_improvement': round(self.mean_improvement, 3),
            'transfer_score': round(self.transfer_score, 3),
            'source_episodes': list(self.source_episodes),
            'parameter_ranges': {
                key: (round(low, 3), round(high, 3))
                for key, (low, high) in self.parameter_ranges.items()
            },
            'categorical_values': {
                key: sorted(values)
                for key, values in self.categorical_values.items()
            },
            'component_types': sorted(self.component_types),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LawSchema':
        schema = cls(
            law_type=str(data.get('law_type', 'unknown')),
            support_count=int(data.get('support_count', 0)),
            source_episodes=list(data.get('source_episodes', [])),
            source_worlds=set(data.get('source_worlds', [])),
            absent_worlds=set(data.get('absent_worlds', [])),
            best_confidence=float(data.get('best_confidence', 0.0)),
            total_improvement=float(data.get('total_improvement', 0.0)),
            categorical_values={
                str(key): set(values)
                for key, values in data.get('categorical_values', {}).items()
            },
            component_types=set(data.get('component_types', [])),
        )
        schema.parameter_ranges = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in data.get('parameter_ranges', {}).items()
        }
        return schema


class LawMemory:
    """
    Stores learned laws across experiments and distills reusable principles.

    This is not yet a hidden policy prior. It is a transparent memory substrate
    that later systems can query before choosing experiments or candidate terms.
    """

    def __init__(self):
        self.episodes: list[ExperimentRecord] = []
        self.schemas: dict[str, LawSchema] = {}

    def record_experiment(
        self,
        world_type: str,
        seed: int,
        object_count: int,
        step_count: int,
        learned_laws: list[dict],
        confirmed_novel_types: set[str] | tuple[str, ...] = (),
        diagnostics: dict | None = None,
    ) -> ExperimentRecord:
        episode_id = f"E_{len(self.episodes) + 1:03d}"
        transfer_report = self.evaluate_transfer(learned_laws)
        record = ExperimentRecord(
            episode_id=episode_id,
            world_type=world_type,
            seed=seed,
            object_count=object_count,
            step_count=step_count,
            learned_laws=list(learned_laws),
            confirmed_novel_types=tuple(sorted(confirmed_novel_types)),
            diagnostics=diagnostics or {},
            transfer_report=transfer_report,
        )
        self.episodes.append(record)
        self._mark_absences(world_type, learned_laws)
        for law in learned_laws:
            self._schema_for(law).update_from_law(episode_id, world_type, law)
        return record

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'episodes': [episode.to_dict() for episode in self.episodes],
            'schemas': {
                law_type: schema.to_dict()
                for law_type, schema in self.schemas.items()
            },
            'principles': self.principles(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LawMemory':
        memory = cls()
        memory.episodes = [
            ExperimentRecord.from_dict(item)
            for item in data.get('episodes', [])
        ]
        memory.schemas = {
            law_type: LawSchema.from_dict(schema_data)
            for law_type, schema_data in data.get('schemas', {}).items()
        }
        return memory

    @classmethod
    def load(cls, path: str | Path) -> 'LawMemory':
        memory_path = Path(path)
        if not memory_path.exists():
            return cls()
        if memory_path.stat().st_size == 0:
            return cls()
        with memory_path.open('r', encoding='utf-8') as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path):
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open('w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    def evaluate_transfer(self, learned_laws: list[dict]) -> dict:
        observed = self._observed_law_types(learned_laws)
        matched = []
        missing = []
        for schema in self.schemas.values():
            item = {
                'law_type': schema.law_type,
                'transfer_score': round(schema.transfer_score, 3),
                'source_worlds': sorted(schema.source_worlds),
            }
            if schema.law_type in observed:
                matched.append(item)
            else:
                missing.append(item)

        matched.sort(key=lambda item: item['transfer_score'], reverse=True)
        missing.sort(key=lambda item: item['transfer_score'], reverse=True)
        return {
            'observed': sorted(observed),
            'matched_priors': matched,
            'missing_priors': missing,
        }

    def suggest_priors(self, limit: int = 5) -> list[dict]:
        priors = [schema.to_dict() for schema in self.schemas.values()]
        priors.sort(key=lambda item: item['transfer_score'], reverse=True)
        return priors[:limit]

    def suggest_probe_questions(self, limit: int = 3) -> list[dict]:
        questions = []
        for schema in self.suggest_priors(limit=limit * 2):
            law_type = schema['law_type']
            if law_type == 'uniform_acceleration':
                continue
            if law_type in ('radial_field', 'inverse_square_radial_field'):
                target = self._center_target(schema)
                question = 'Does a prior radial field transfer to this world, or is it absent here?'
            elif law_type == 'tangential_field':
                target = self._center_target(schema)
                question = 'Does a prior tangential field transfer to this world, or is it absent here?'
            elif law_type == 'time_varying_field':
                target = {'kind': 'center'}
                question = 'Does this world contain the prior temporal force pattern?'
            else:
                target = {'kind': 'center'}
                question = f"Does prior law family {law_type} transfer to this world?"
            questions.append({
                'law_type': law_type,
                'question': question,
                'transfer_score': schema['transfer_score'],
                'target': target,
            })
            if len(questions) >= limit:
                break
        return questions

    def principles(self) -> list[str]:
        if not self.schemas:
            return []

        principles = []
        if self._has_schema('uniform_acceleration'):
            principles.append("Some worlds contain persistent uniform acceleration components.")
        if self._has_schema('radial_field') or self._has_schema('inverse_square_radial_field'):
            principles.append("Fields can be centered on points and act radially toward or away from them.")
        if self._has_schema('tangential_field'):
            principles.append("Fields can rotate motion around a center as a tangential law.")
        if self._has_schema('time_varying_field'):
            principles.append("Forces can vary over time and are sometimes better described by periods.")
        if self._has_schema('composed_dynamics'):
            principles.append("World dynamics can combine multiple simpler law families into one explanation.")
        return principles

    def summary(self) -> str:
        lines = [
            "Cross-experiment law memory:",
            f"  Episodes: {len(self.episodes)}",
            f"  Law schemas: {len(self.schemas)}",
        ]
        priors = self.suggest_priors()
        if priors:
            lines.append("  Strongest reusable schemas:")
            for schema in priors:
                worlds = ','.join(schema['source_worlds']) or 'none'
                lines.append(
                    f"    {schema['law_type']}: support={schema['support_count']}, "
                    f"score={schema['transfer_score']:.2f}, worlds={worlds}"
                )
        principles = self.principles()
        if principles:
            lines.append("  Consolidated principles:")
            for principle in principles:
                lines.append(f"    - {principle}")
        return "\n".join(lines)

    def install_principles(self, knowledge_base, step: int):
        existing = {
            concept.properties.get('memory_principle')
            for concept in knowledge_base.get_meta_concepts()
        }
        for idx, principle in enumerate(self.principles(), start=1):
            if principle in existing:
                continue
            knowledge_base.add_meta_concept(
                name=f"LM_{idx:03d}",
                description=principle,
                sub_concept_names=[],
                step=step,
                properties={
                    'source': 'law_memory',
                    'memory_principle': principle,
                    'schema_count': len(self.schemas),
                    'episode_count': len(self.episodes),
                },
            )

    def _schema_for(self, law: dict) -> LawSchema:
        law_type = str(law.get('law_type', 'unknown'))
        if law_type not in self.schemas:
            self.schemas[law_type] = LawSchema(law_type=law_type)
        return self.schemas[law_type]

    def _mark_absences(self, world_type: str, learned_laws: list[dict]):
        observed = self._observed_law_types(learned_laws)
        for schema in self.schemas.values():
            if schema.law_type not in observed and world_type not in schema.source_worlds:
                schema.absent_worlds.add(world_type)

    def _observed_law_types(self, learned_laws: list[dict]) -> set[str]:
        observed = set()
        for law in learned_laws:
            law_type = str(law.get('law_type', 'unknown'))
            observed.add(law_type)
            for component in law.get('properties', {}).get('components', []):
                observed.add(str(component))
        return observed

    def _has_schema(self, law_type: str) -> bool:
        return law_type in self.schemas and self.schemas[law_type].support_count > 0

    def _center_target(self, schema: dict) -> dict:
        ranges = schema.get('parameter_ranges', {})
        if 'center_x' not in ranges or 'center_y' not in ranges:
            return {'kind': 'center'}
        x_low, x_high = ranges['center_x']
        y_low, y_high = ranges['center_y']
        return {
            'kind': 'point',
            'x': (x_low + x_high) / 2,
            'y': (y_low + y_high) / 2,
        }
