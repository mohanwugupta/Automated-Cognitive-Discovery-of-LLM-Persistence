"""Prospective fixed-controller stability and matched-baseline experiment.

Run from the repository root. All fitting uses original train pairs; fresh
contrasts are evaluation-only. Writes a protocol before loading model weights.
"""
from pathlib import Path
import argparse, dataclasses, hashlib, json, time
import numpy as np
import pandas as pd
from cognitive_discovery.causal_mechanistic import pipeline as p
from cognitive_discovery.causal_mechanistic.das import DASAlignment, save_alignment, load_alignment
from cognitive_discovery.causal_mechanistic.counterfactuals import FrozenTheoryBank, build_counterfactual_pairs
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.mechanistic.dataset.matched_conditions import load_mechanistic_manifest, _history
from cognitive_discovery.mechanistic.targets.behavioral_targets import compute_condition_targets


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def fresh_design(records, pairs):
    """Four signed history changes per task/mapping; no neural outcome selection."""
    patterns = [((-1,-1,-1), (1,-1,-1)), ((-1,-1,-1), (1,1,-1)),
                ((1,1,1), (-1,1,1)), ((1,1,1), (-1,-1,1))]
    by_id = {r.condition.condition_id:r for r in records}
    result=[]
    eligible=pairs[pairs.pair_split.isin(['mech_pair_test','mech_task_holdout'])]
    for (task,mapping), group in eligible.groupby(['task_family','response_mapping']):
        template=by_id[group.sort_values('pair_id').iloc[0].base_condition_id]
        for i,(left,right) in enumerate(patterns):
            cid=f'prospective_history_{task}_{mapping}_{i}'
            for member,outcomes in [(-1,left),(1,right)]:
                c=dataclasses.replace(template.condition,condition_id=f'{cid}_{member}',
                    history=_history(outcomes,template.condition.history.actions))
                result.append(dataclasses.replace(template,condition=c,contrast_id=cid,
                    contrast_member=member,targets=compute_condition_targets(c)))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='artifacts/qwen_runpod_v1')
    ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--revision',required=True); ap.add_argument('--random-count',type=int,default=20)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    root=Path('artifacts/causal_mech_v1')
    job=json.loads((root/'representations/selected_alignments.json').read_text())[1]
    pairs=p._job_pairs(root,job,('mech_pair_train','mech_pair_validation','mech_pair_test','mech_task_holdout'))
    records=load_mechanistic_manifest(root/'counterfactuals/mechanistic_conditions.jsonl')
    fresh=fresh_design(records,pairs)
    bank=FrozenTheoryBank(Path('artifacts/theory_resolution_v1/frozen_models'))
    fp, predictions=build_counterfactual_pairs(fresh,bank)
    fp=fp[fp.target_variable.eq('outcome_history')].copy()
    preds=p._prediction_lookup(root,'dual_history')
    preds=pd.concat([preds,predictions[predictions.theory.eq('dual_history')].set_index('pair_id').predicted_counterfactual_effect])
    # Validate numerical target variation before any neural evaluation.
    target_audit=[]
    for (task,mapping),group in fp.groupby(['task_family','response_mapping']):
        values=preds.loc[group.pair_id].to_numpy()
        assert np.ptp(values)>1e-6, (task,mapping,values)
        assert (values<0).any() and (values>0).any()
        target_audit.append(dict(task=task,mapping=mapping,targets=values.tolist()))
    fp['evaluation_set']='fresh_signed_history'
    pairs['evaluation_set']='original_pairs'
    all_pairs=pd.concat([pairs,fp],ignore_index=True)
    all_pairs.to_parquet(out/'pairs.parquet',index=False)
    predictions.to_parquet(out/'fresh_predictions.parquet',index=False)
    (out/'fresh_conditions.jsonl').write_text('\n'.join(json.dumps(r.to_dict()) for r in fresh)+'\n')
    protocol=dict(model='Qwen/Qwen3.5-4B',revision=args.revision,layer=28,rank=2,
        seeds=[75001,75002,75003,75004,75005],epochs=3,lr=.03,activation_penalty=1e-4,
        fitting='original mech_pair_train only; no new hyperparameter selection',
        evaluation='all five seeds reported separately; original validation/test/holdout and fresh signed histories',
        baselines=['ridge_history_rank2','mean_difference_rank2','pca_rank2','ridge_persistence_rank2','direct_DAS_rank2'],
        matching='same layer/rank/evaluation pairs; baseline intervention norms matched to frozen original DAS',
        ridge_alpha=1.0,random_count=args.random_count,random_seed=76001,
        fresh_target_audit=target_audit,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        original_revision_unknown=True,scope='fixed primary dual-history controller; other frozen controllers not rerun',
        max_wall_seconds=10800)
    dump(out/'protocol.json',protocol)
    if args.prepare_only: return
    records_by_id={r.condition.condition_id:r for r in records+fresh}
    import torch
    from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
    from cognitive_discovery.causal_mechanistic.interventions import interchange_subspace
    print('PROTOCOL_SAVED',flush=True)
    runner=MechanisticQwenRunner.from_pretrained(protocol['model'],revision=args.revision,local_files_only=True)
    for param in runner.model.parameters():param.requires_grad_(False)
    print('MODEL_LOADED',flush=True)
    cache={}; ids=sorted(set(all_pairs.base_condition_id)|set(all_pairs.source_condition_id))
    for i,cid in enumerate(ids):
        result=p._forward(runner,records_by_id[cid],capture_layers=(28,))
        cache[cid]=result
        if i%20==0:print('CACHE',i,len(ids),flush=True)
    states={cid:r.states[28] for cid,r in cache.items()}
    baseline=pd.DataFrame([dict(pair_id=r.pair_id,base_persistence_logit=cache[r.base_condition_id].persistence_logit)
        for r in all_pairs.itertuples()]).set_index('pair_id')
    baseline.to_parquet(out/'fresh_baselines.parquet')
    np.savez_compressed(out/'decision_states.npz',**states)
    old=p._localization_lookup(root,28,'dual_history')
    diffs=baseline.loc[pairs.pair_id].base_persistence_logit-old.loc[pairs.pair_id].base_persistence_logit
    dump(out/'reproduction_audit.json',dict(max_abs_baseline_delta=float(diffs.abs().max()),mean_abs_baseline_delta=float(diffs.abs().mean()),
        note='Original artifacts did not record an immutable model revision; all new effects use recomputed baselines.'))
    train=pairs[pairs.pair_split.eq('mech_pair_train')]
    eval_pairs=all_pairs[~all_pairs.pair_split.eq('mech_pair_train')]
    assert not set(train.pair_id)&set(eval_pairs.pair_id)
    frozen=load_alignment(job['artifact']); bases={'frozen_DAS':frozen}; summaries=[]
    norms={r.pair_id:float(np.linalg.norm((states[r.source_condition_id]-states[r.base_condition_id])@frozen)) for r in eval_pairs.itertuples()}
    def evaluate(name,basis,match=False):
        path=out/'evaluations'/f'{name}.parquet'; path.parent.mkdir(exist_ok=True)
        rows=p._evaluate_alignment(runner,basis,eval_pairs,records_by_id,states,baseline,preds,layer=28,
            split_label='see_pair_manifest',intervention_type=name,matched_norms=norms if match else None)
        df=pd.DataFrame(rows).drop(columns='pair_split').merge(eval_pairs[['pair_id','pair_split','evaluation_set']],on='pair_id',validate='one_to_one')
        df.to_parquet(path,index=False)
        for (split,dataset),g in df.groupby(['pair_split','evaluation_set']):
            summaries.append(dict(method=name,split=split,dataset=dataset,n=len(g),**counterfactual_metrics(g.predicted_counterfactual_effect,g.neural_counterfactual_effect)))
        pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False)
        print('EVALUATED',name,flush=True)
    evaluate('frozen_DAS',frozen)
    def fit(seed,direct=False):
        a=DASAlignment(len(next(iter(states.values()))),2,seed=seed+2,device=next(runner.model.parameters()).device)
        opt=torch.optim.Adam(a.parameters(),lr=.03); losses=[]
        for epoch in range(3):
            epoch_losses=[]
            for pair in train.sample(frac=1,random_state=seed+epoch).itertuples():
                penalty={}; opt.zero_grad(set_to_none=True)
                def editor(state):
                    edited=a.edit(state,states[pair.source_condition_id]); penalty['v']=(edited-state).float().square().mean();return edited
                rec=records_by_id[pair.base_condition_id]; trial=p._trial(rec)
                observed=runner.differentiable_persistence_logit(list(trial.messages),rec.condition.response_mapping.labels,
                    positive_label=rec.condition.response_mapping.continue_label,editors={28:editor})
                effect=(cache[pair.source_condition_id].persistence_logit-cache[pair.base_condition_id].persistence_logit) if direct else preds.loc[pair.pair_id]
                loss=(observed.float()-baseline.loc[pair.pair_id,'base_persistence_logit']-effect).square()+1e-4*penalty['v']+2e-5
                loss.backward();assert torch.isfinite(a.parameter.grad).all();opt.step();epoch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_losses))); print('TRAIN',seed,direct,epoch,losses[-1],flush=True)
        basis=a.numpy_basis();name=f'direct_DAS_{seed}' if direct else f'cognitive_DAS_{seed}'
        save_alignment(out/'alignments'/f'{name}.safetensors',basis,metadata=dict(seed=seed,direct=direct,losses=losses,train_pairs=train.pair_id.tolist()))
        return name,basis
    for seed in protocol['seeds']:
        name,basis=fit(seed);evaluate(name,basis)
    name,basis=fit(protocol['seeds'][0],direct=True);evaluate(name,basis,True)
    # All conventional directions fit to original train condition states only.
    train_ids=sorted(set(train.base_condition_id)|set(train.source_condition_id))
    X=np.vstack([states[c] for c in train_ids]); centered=X-X.mean(0)
    from sklearn.linear_model import Ridge
    y=np.array([compute_condition_targets(records_by_id[c].condition)['outcome_history'] for c in train_ids])
    persistence=np.array([cache[c].persistence_logit for c in train_ids])
    _,_,vt=np.linalg.svd(centered,full_matrices=False)
    def rank2(vector):
        columns=[vector]+list(vt[:3]); q=[]
        for col in columns:
            col=col.copy()
            for v in q: col-=v*np.dot(v,col)
            if np.linalg.norm(col)>1e-8:q.append(col/np.linalg.norm(col))
            if len(q)==2:break
        return np.column_stack(q).astype(np.float32)
    bases={'ridge_history_rank2':rank2(Ridge(alpha=1).fit(X,y).coef_),
        'mean_difference_rank2':rank2(X[y>np.median(y)].mean(0)-X[y<=np.median(y)].mean(0)),
        'pca_rank2':vt[:2].T.astype(np.float32),
        'ridge_persistence_rank2':rank2(Ridge(alpha=1).fit(X,persistence).coef_)}
    for name,basis in bases.items():
        save_alignment(out/'alignments'/f'{name}.safetensors',basis,metadata={'fitting_split':'mech_pair_train','rank_completion':'orthogonal train PCA when scalar target supplies one direction'})
        evaluate(name,basis,True)
    rng=np.random.default_rng(protocol['random_seed'])
    for i in range(args.random_count):
        basis=np.linalg.qr(rng.normal(size=frozen.shape),mode='reduced')[0].astype(np.float32)
        evaluate(f'random_{i:03d}',basis,True)
    dump(out/'complete.json',{'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
