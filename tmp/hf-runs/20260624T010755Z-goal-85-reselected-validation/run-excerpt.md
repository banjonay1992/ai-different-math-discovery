# HF goal-85 reselected validation

- Run id: `goal-85-reselected-validation-ba20098-1782262430`
- Commit: `ba20098a60a661a1b3fc30ad05ef59160e8d0f0c`
- Runs final: `false`
- Progress: `79.8% -> 77.7%`
- Target reached: `false`

## Result rows
- 1. `selected_law_replay` in `repulsion` seed `13` -> `selected_law_replay_conflicted`; equation `k * taper(separation, 6_065) * unit_generated_center_vector / separation^0_5` score `0.914`
- 2. `selected_law_conflict_resolution` in `repulsion` seed `14` -> `conflict_domain_split_supported`; equation `k * taper(separation, 6_065) * unit_generated_center_vector / separation^1_25` score `0.898`
- 3. `blind_holdout_validation` in `hidden_05_0005` seed `5` -> `blind_holdout_absent`; equation `k * taper(separation, 7_576) * unit_generated_center_vector / separation^1_5` score `0.854`
- 4. `blind_holdout_validation` in `repulsion` seed `15` -> `blind_holdout_conflicted`; equation `k * taper(separation, 6_065) * unit_generated_center_vector / separation^1` score `0.914`
- 5. `selected_law_conflict_resolution` in `repulsion` seed `16` -> `conflict_domain_split_supported`; equation `k * taper(separation, 5_907) * unit_generated_center_vector / separation^1` score `0.888`

## Read

The selected repulsion law kept alternating. The useful signal is repeated `conflict_domain_split_supported`, so the next code change should prioritize explicit predicate/piecewise law work over replaying a single universal exponent.
