"""Frozen-controller evaluation on fully counterbalanced fresh contexts.

All three retained controllers; four backgrounds/task and four signed patterns.
No fitting, layer/rank selection, or outcome-dependent selection occurs here.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from cognitive_discovery.pipeline import generate_design, load_config
from cognitive_discovery.design.counterbalance import counterbalanced_mappings
from cognitive_discovery.mechanistic.dataset.matched_conditions import (
    MatchedMechanisticCondition, _history, _context, validate_matched_conditions,
    load_mechanistic_manifest)
from cognitive_discovery.mechanistic.targets.behavioral_targets import compute_condition_targets
from cognitive_discovery.causal_mechanistic.counterfactuals import FrozenTheoryBank, build_counterfactual_pairs
from cognitive_discovery.causal_mechanistic.das import load_alignment, save_alignment
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.causal_mechanistic import pipeline as p


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def prepare(out, revision, *, seed=87001, patterns_override=None):
    config = load_config('configs/discovery_v1.yaml')
    design, _ = generate_design(config, output=out/'backgrounds', conditions=490,
                                seed=seed, design_id='qwen_fresh_contexts_v1')
    patterns = [((-1,-1,-1),(1,-1,-1)), ((1,1,1),(-1,-1,1)),
                ((-1,-1,-1,-1,-1),(1,1,-1,-1,-1)),
                ((1,1,1,1,1),(-1,-1,-1,1,1))]
    if patterns_override is not None: patterns = patterns_override
    heldout = {'waiting','effort','information_sampling'}
    records=[]
    for task in sorted({c.task_family for c in design}):
        candidates=sorted([c for c in design if c.task_family==task and c.response_mapping.mapping_id=='continue_x'],
                          key=lambda c:c.condition_id)
        backgrounds=[]; seen=set()
        for c in candidates:
            key=json.dumps(c.semantic_factors,sort_keys=True)
            if key not in seen:
                backgrounds.append(c);seen.add(key)
            if len(backgrounds)==4:break
        assert len(backgrounds)==4
        for b, template in enumerate(backgrounds):
            for pat,(left,right) in enumerate(patterns):
                actions=tuple(('continue','disengage','continue','continue','disengage')[:len(left)])
                families=['outcome_history']+(['contextual_history'] if task in {'bandit','foraging','debugging'} else [])
                for family in families:
                    cid=f'fresh_contexts:{task}:bg{b}:pattern{pat}:{family}'
                    for member, outcomes in [(-1,left),(1,right)]:
                        if family=='outcome_history':
                            history=_history(outcomes,actions);context=None;target='outcome_history'
                        else:
                            history=_history(left,actions)
                            context=_context(contrast_id=cid,a_history=_history(right,actions),
                                             b_history=history,context_return='A' if member==1 else 'B')
                            target='contextual_outcome_history'
                        pid=f'{cid}:member{member}'
                        for m,mapping in enumerate(counterbalanced_mappings(('X','Y'))):
                            c=dataclasses.replace(template,design_id='qwen_fresh_contexts_v1',
                                condition_id=f'{pid}-m{m}',paired_condition_id=pid,
                                response_mapping=mapping,history=history,contextual_history=context,
                                split='transfer' if task in heldout else 'test',sampling_strategy=family)
                            records.append(MatchedMechanisticCondition(c,cid,family,member,target,compute_condition_targets(c)))
    # The standard manifest uses increasing-target signed members. Evaluation
    # directions retain the prespecified positive AND negative history changes.
    oriented_records=records
    records=[dataclasses.replace(r,contrast_member=-r.contrast_member)
             if ':pattern1:' in r.contrast_id or ':pattern3:' in r.contrast_id else r
             for r in records]
    validate_matched_conditions(records)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'conditions.jsonl'
    manifest.write_text(''.join(json.dumps(r.to_dict())+'\n' for r in records))
    assert len(load_mechanistic_manifest(manifest))==len(records)
    bank=FrozenTheoryBank(Path('artifacts/theory_resolution_v1/frozen_models'))
    pairs,predictions=build_counterfactual_pairs(oriented_records,bank)
    pairs['background']=pairs.contrast_id.str.extract(r':bg(\d+):').astype(int)
    pairs['pattern']=pairs.contrast_id.str.extract(r':pattern(\d+):').astype(int)
    pairs.to_parquet(out/'pairs.parquet',index=False)
    predictions.to_parquet(out/'predictions.parquet',index=False)
    jobs=json.loads(Path('artifacts/causal_mech_v1/representations/selected_alignments.json').read_text())
    protocol=dict(model='Qwen/Qwen3.5-4B',revision=revision,seed=seed,
                  backgrounds_per_task=4,patterns=patterns,conditions=len(records),pairs=len(pairs),
                  controllers=jobs,random_count=20,random_seed=87002,
                  intervention='source-to-base subspace interchange at final prompt token; explicit pair table sets signed orientation, manifest members are increasing-target order',
                  random_matching='same layer/rank; match frozen-controller norm for each pair',
                  fitting='none; all historical controllers and cognitive models frozen',
                  scope='fresh current-state contexts and length-3/5 histories; existing renderer wording',
                  inference='report all three controllers; paired background-level bootstrap and random null',
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  theory_hashes=bank.hashes,
                  alignment_hashes={j['artifact']:hashlib.sha256(Path(j['artifact']).read_bytes()).hexdigest() for j in jobs})
    dump(out/'protocol.json',protocol)
    return records,pairs,predictions,jobs,protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    out=Path('artifacts/qwen_fresh_contexts_v1')
    records,pairs,predictions,jobs,protocol=prepare(out,args.revision)
    print('PROTOCOL',len(records),len(pairs),flush=True)
    if args.prepare_only:return
    import torch,transformers
    from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
    start=time.monotonic()
    runner=MechanisticQwenRunner.from_pretrained(protocol['model'],revision=args.revision)
    record_map={r.condition.condition_id:r for r in records}
    cache={}
    for i,(cid,record) in enumerate(record_map.items()):
        cache[cid]=p._forward(runner,record,capture_layers=(28,30))
        if i%40==0:print('CACHE',i,len(records),flush=True)
    baselines=pd.DataFrame([dict(pair_id=r.pair_id,base_persistence_logit=cache[r.base_condition_id].persistence_logit)
                            for r in pairs.itertuples()]).set_index('pair_id')
    baselines.to_parquet(out/'baselines.parquet')
    summary=[]
    for j,job in enumerate(jobs):
        selected=pairs[pairs.target_variable.eq(job['target_variable'])].copy()
        # Context controller discovery uses context-isolated tasks; raw history is its held-out boundary.
        if job['target_variable']=='contextual_outcome_history':
            selected=selected[selected.counterfactual_subtype.isin(['context_isolated','heldout_broad_history'])]
        states={cid:r.states[job['layer']] for cid,r in cache.items()}
        basis=load_alignment(job['artifact'])
        pred=predictions[predictions.theory.eq(job['theory'])].set_index('pair_id').predicted_counterfactual_effect
        norms={r.pair_id:float(np.linalg.norm((states[r.source_condition_id]-states[r.base_condition_id])@basis))
               for r in selected.itertuples()}
        rng=np.random.default_rng(protocol['random_seed']+j)
        for k in range(protocol['random_count']+1):
            name='frozen' if k==0 else f'random_{k-1:03d}'
            intervention=basis if k==0 else np.linalg.qr(rng.normal(size=basis.shape),mode='reduced')[0].astype(np.float32)
            save_alignment(out/'bases'/f'{job["theory"]}_{name}.safetensors',intervention,metadata={'method':name})
            rows=p._evaluate_alignment(runner,intervention,selected,record_map,states,baselines,pred,
                    layer=job['layer'],split_label='see_pair_manifest',intervention_type=name,
                    matched_norms=None if k==0 else norms)
            frame=pd.DataFrame(rows).drop(columns='pair_split').merge(
                selected[['pair_id','pair_split','background','pattern']],on='pair_id',validate='one_to_one')
            path=out/'evaluations'/f'{job["theory"]}_{name}.parquet';path.parent.mkdir(exist_ok=True)
            frame.to_parquet(path,index=False)
            for split,g in frame.groupby('pair_split'):
                summary.append(dict(theory=job['theory'],method=name,split=split,n=len(g),
                    **counterfactual_metrics(g.predicted_counterfactual_effect,g.neural_counterfactual_effect)))
            pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
            print('EVALUATED',job['theory'],name,len(frame),flush=True)
    dump(out/'complete.json',dict(seconds=time.monotonic()-start,torch=torch.__version__,
             transformers=transformers.__version__,peak_vram_bytes=torch.cuda.max_memory_allocated(),
             evaluation_rows=sum(r['n'] for r in summary)))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
