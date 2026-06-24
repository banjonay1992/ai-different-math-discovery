# HF run goal-85-prioritized-blind-927dc94-1782253777
job_id: 6a3b08cc3fe5564453284fa0
commit: 927dc94b983b04c8166faea59fe39bfc1f7933c6
starting_progress: 79.8
ending_progress: 77.7
target_reached: False

## Rows
- 1: post_run_replay_revision hidden_03_0003 seed=0 -> replay_still_baseline_headline score=0.97 expr=x + vx * dt
- 2: blind_holdout_validation repulsion seed=9 -> blind_holdout_conflicted score=0.914 expr=k * taper(separation, 6_065) * unit_generated_center_vector / separation^1
- 3: blind_holdout_validation inverse_square_repulsion seed=8 -> blind_holdout_conflicted score=0.811 expr=k * unit_local_inferred_vector / separation^1_5
- 4: blind_holdout_validation repulsion seed=10 -> blind_holdout_conflicted score=0.905 expr=k * taper(separation, 6_065) * unit_generated_center_vector / separation^1_75
- 5: blind_holdout_validation inverse_square_repulsion seed=9 -> blind_holdout_conflicted score=0.874 expr=k * taper(separation, 9_067) * unit_generated_center_vector / separation^0_5
