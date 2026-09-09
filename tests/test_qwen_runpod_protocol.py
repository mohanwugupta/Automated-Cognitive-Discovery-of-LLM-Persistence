"""Scientific invariants for the prospective RunPod design."""
import importlib.util
from pathlib import Path
import json
import numpy as np
from cognitive_discovery.causal_mechanistic import pipeline as p
from cognitive_discovery.causal_mechanistic.counterfactuals import FrozenTheoryBank, build_counterfactual_pairs
from cognitive_discovery.mechanistic.dataset.matched_conditions import load_mechanistic_manifest


def test_fresh_histories_preserve_split_and_have_informative_signed_targets():
    spec=importlib.util.spec_from_file_location('qwen_runpod',Path(__file__).parents[1]/'scripts/qwen_runpod_experiment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=Path('artifacts/causal_mech_v1')
    job=json.loads((root/'representations/selected_alignments.json').read_text())[1]
    pairs=p._job_pairs(root,job,('mech_pair_train','mech_pair_validation','mech_pair_test','mech_task_holdout'))
    records=load_mechanistic_manifest(root/'counterfactuals/mechanistic_conditions.jsonl')
    fresh=module.fresh_design(records,pairs)
    original_ids={r.condition.condition_id for r in records}
    assert len(fresh)==112
    assert not original_ids & {r.condition.condition_id for r in fresh}
    assert {r.condition.split for r in fresh}=={'test','transfer'}
    fp,targets=build_counterfactual_pairs(fresh,FrozenTheoryBank('artifacts/theory_resolution_v1/frozen_models'))
    selected=targets[(targets.theory=='dual_history')&(targets.target_variable=='outcome_history')]
    joined=fp.merge(selected[['pair_id','predicted_counterfactual_effect']],on='pair_id')
    assert len(joined)==56
    for _,g in joined.groupby(['task_family','response_mapping']):
        y=g.predicted_counterfactual_effect.to_numpy()
        assert np.ptp(y)>1e-6
        assert len(np.unique(np.round(y,8)))==4
        assert (y<0).sum()==(y>0).sum()==2
        assert np.all(np.abs(y-np.roll(y,1))>1e-6)
