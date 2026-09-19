"""Descriptive CPU audit of existing abstraction contrasts; no new inference."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = {k: f'predicted_{k}_effect' for k in ('O', 'O_star', 'H', 'E')}
PRIMARY = 'das_001_outcome_history_dual_history_L28_r2'

def pair_diagnostics(frame):
    rows = []
    for (a, ac), (b, bc) in itertools.combinations(TARGETS.items(), 2):
        x, y = frame[ac].to_numpy(), frame[bc].to_numpy()
        active = (np.abs(x) > .1) & (np.abs(y) > .1)
        rows.append(dict(left=a, right=b, rendered_pairs=len(frame),
            correlation=float(np.corrcoef(x, y)[0, 1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else None,
            rms_difference=float(np.sqrt(np.mean((x-y)**2))),
            equal_fraction=float(np.mean(np.isclose(x, y, atol=1e-10, rtol=0))),
            active_pairs=int(active.sum()),
            opposite_sign_active_pairs=int(((x*y < 0) & active).sum())))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('artifacts/abstraction_discovery_v1'))
    p.add_argument('--output', type=Path, default=Path('paper/generated/computational_discrimination'))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    paths = [a.root/'design/contrast_manifest.parquet', a.root/'frozen_das/interchange_results.parquet']
    design, neural = map(pd.read_parquet, paths)
    primary = neural[neural.artifact_id == PRIMARY].copy()
    assert not design.pair_id.duplicated().any()
    assert not primary.pair_id.duplicated().any()
    expected = design[design.pair_split == 'abstraction_test']
    assert set(primary.pair_id) == set(expected.pair_id), 'Incomplete or unexpected final evaluation'
    aligned = primary.set_index('pair_id').loc[expected.pair_id]
    for c in TARGETS.values():
        np.testing.assert_allclose(aligned[c], expected[c], rtol=0, atol=1e-12)
    assert (primary.groupby('semantic_contrast_id').response_mapping.nunique() == 2).all()
    rows = []
    for name, d in [('all_design', design), ('evaluated_test', primary)]:
        for group, f in [('all', d), *list(d.groupby('contrast_family'))]:
            rows.extend(dict(scope=name, family=group, **r) for r in pair_diagnostics(f))
    pd.DataFrame(rows).to_csv(a.output/'target_discrimination.csv', index=False)
    family = []
    for name, f in primary.groupby('contrast_family'):
        y = f.neural_counterfactual_effect.to_numpy()
        record = dict(family=name, semantic_pairs=f.semantic_contrast_id.nunique(),
                      rendered_pairs=len(f), tasks=f.task_family.nunique())
        for k, c in TARGETS.items():
            x = f[c].to_numpy(); den = float(x @ x)
            record[f'{k}_target_energy'] = den
            record[f'{k}_rmse'] = float(np.sqrt(np.mean((y-x)**2)))
            record[f'{k}_recovery'] = float(1-((y-x)@(y-x))/den) if den>1e-12 else None
        family.append(record)
    families = pd.DataFrame(family)
    families['E_target_energy_fraction'] = families.E_target_energy / families.E_target_energy.sum()
    families.to_csv(a.output/'family_recovery.csv', index=False)
    used=set(primary.base_semantic_id)|set(primary.source_semantic_id)
    split_overlap={}
    for split, f in design.groupby('pair_split'):
        states=set(f.base_semantic_id)|set(f.source_semantic_id)
        split_overlap[split]=len(used & states)
    summary=dict(status='descriptive_post_hoc_audit_not_confirmatory',
        design_rendered_pairs=len(design), design_semantic_pairs=design.semantic_contrast_id.nunique(),
        evaluated_rendered_pairs=len(primary), evaluated_semantic_pairs=primary.semantic_contrast_id.nunique(),
        evaluated_unique_states=len(used), test_state_overlap_by_design_split=split_overlap,
        evaluated_rows_match_declared_test=True,
        thresholds={'equal_atol':1e-10,'active_absolute_logit':.1},
        all_target_pairs=pair_diagnostics(primary),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (a.output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
