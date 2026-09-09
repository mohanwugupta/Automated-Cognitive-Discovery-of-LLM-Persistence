"""Replay fresh-context effects, clustered uncertainty, and matched nulls."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics,global_counterfactual_recovery


def main():
    root=Path('artifacts/qwen_fresh_contexts_v1')
    out=Path('paper/generated/qwen_fresh_contexts');out.mkdir(parents=True,exist_ok=True)
    files=sorted((root/'evaluations').glob('*.parquet'))
    if files:
        frames=[]
        for path in files:
            frame=pd.read_parquet(path)
            theory,method=path.stem.split('_frozen')[0],'frozen'
            if '_random_' in path.stem:
                theory,index=path.stem.rsplit('_random_',1);method=f'random_{index}'
            frame['theory']=theory;frame['method']=method;frames.append(frame)
        data=pd.concat(frames,ignore_index=True)
        data.to_csv(out/'interventions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pd.read_parquet(root/'pairs.parquet').to_csv(out/'pairs.csv',index=False)
        for filename in ['protocol.json','complete.json','resource_audit.json']:
            if (root/filename).exists():shutil.copy2(root/filename,out/filename)
    else:
        data=pd.read_csv(out/'interventions.csv.gz')
    assert len(data)==13440,len(data)
    assert not data.duplicated(['theory','method','pair_id']).any()
    assert data.groupby(['theory','method']).size().groupby(level=0).nunique().eq(1).all()
    summaries=[]
    for (theory,method,split),g in data.groupby(['theory','method','pair_split']):
        summaries.append(dict(theory=theory,method=method,split=split,n=len(g),
            **counterfactual_metrics(g.predicted_counterfactual_effect,g.neural_counterfactual_effect)))
    metrics=pd.DataFrame(summaries);metrics.to_csv(out/'metrics.csv',index=False)
    comparisons=[];intervals=[];taskrows=[];normrows=[]
    for (theory,split),g in data.groupby(['theory','pair_split']):
        sub=metrics[metrics.theory.eq(theory)&metrics.split.eq(split)]
        score=float(sub[sub.method.eq('frozen')].global_cfr.iloc[0])
        null=sub[sub.method.str.startswith('random_')].global_cfr.to_numpy()
        assert len(null)==20
        comparisons.append(dict(theory=theory,split=split,frozen_global_cfr=score,
            random_mean=float(null.mean()),random_max=float(null.max()),
            p_plus_one=float((1+(null>=score).sum())/(1+len(null)))))
        frozen=g[g.method.eq('frozen')].reset_index(drop=True)
        strata={task:[bg.index.to_numpy() for _,bg in part.groupby('background')]
                for task,part in frozen.groupby('task_family')}
        rng=np.random.default_rng(87003);boot=[]
        for _ in range(1000):
            indices=np.concatenate([np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
                                    for groups in strata.values()])
            b=frozen.iloc[indices]
            boot.append(global_counterfactual_recovery(b.predicted_counterfactual_effect,b.neural_counterfactual_effect))
        intervals.append(dict(theory=theory,split=split,global_cfr=score,
            ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
            bootstrap='1000 resamples of background clusters within task; all histories/mappings stay together',
            clusters=sum(len(v) for v in strata.values())))
        for task,t in frozen.groupby('task_family'):
            taskrows.append(dict(theory=theory,split=split,task=task,n=len(t),
                **counterfactual_metrics(t.predicted_counterfactual_effect,t.neural_counterfactual_effect)))
        norm=frozen.set_index('pair_id').intervention_norm
        for method,random in g[g.method.str.startswith('random_')].groupby('method'):
            ref=norm.loc[random.pair_id].to_numpy();observed=random.intervention_norm.to_numpy()
            rel=np.abs(observed-ref)/np.maximum(ref,1e-12)
            normrows.append(dict(theory=theory,split=split,method=method,
                                 median_relative_error=float(np.median(rel)),max_relative_error=float(rel.max())))
    for name,rows in [('random_comparisons',comparisons),('bootstrap',intervals),('tasks',taskrows),('norm_audit',normrows)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    natural_path=root/'natural_effects.parquet'
    if natural_path.exists():
        natural=pd.read_parquet(natural_path)
        natural.to_csv(out/'natural_effects.csv',index=False)
        shutil.copy2(root/'natural_effects_protocol.json',out/'natural_effects_protocol.json')
    elif (out/'natural_effects.csv').exists():
        natural=pd.read_csv(out/'natural_effects.csv')
    else:
        natural=None
    if natural is not None:
        joined=data[data.method.eq('frozen')].merge(natural[['pair_id','natural_effect']],on='pair_id',validate='many_to_one')
        calibration=[]
        for (theory,split),g in joined.groupby(['theory','pair_split']):
            calibration.append(dict(theory=theory,split=split,n=len(g),
                **counterfactual_metrics(g.predicted_counterfactual_effect,g.natural_effect)))
        pd.DataFrame(calibration).to_csv(out/'natural_calibration.csv',index=False)
        # Secondary diagnostic: a weak cognitive target need not imply poor
        # recovery of the model's own natural source-versus-base response.
        joined_all=data.merge(natural[['pair_id','natural_effect']],on='pair_id',validate='many_to_one')
        natural_recovery=[]
        for (theory,method,split),g in joined_all.groupby(['theory','method','pair_split']):
            natural_recovery.append(dict(theory=theory,method=method,split=split,n=len(g),
                analysis_role='secondary natural-effect diagnostic; not original cognitive-target endpoint',
                **counterfactual_metrics(g.natural_effect,g.neural_counterfactual_effect)))
        pd.DataFrame(natural_recovery).to_csv(out/'natural_effect_recovery.csv',index=False)
        sensitivity=[]
        for (theory,split),g in joined_all.groupby(['theory','pair_split']):
            frozen=g[g.method.eq('frozen')].set_index('pair_id')
            null=g[g.method.str.startswith('random_')]
            norms=null.pivot(index='pair_id',columns='method',values='intervention_norm').reindex(frozen.index)
            error=norms.sub(frozen.intervention_norm,axis=0).abs().div(frozen.intervention_norm.clip(lower=1e-12),axis=0)
            ids=frozen.index[error.max(axis=1).le(.05)]
            if not len(ids):continue
            for endpoint,column in [('cognitive','predicted_counterfactual_effect'),('natural','natural_effect')]:
                target=frozen.loc[ids,column]
                score=global_counterfactual_recovery(target,frozen.loc[ids].neural_counterfactual_effect)
                scores=[global_counterfactual_recovery(target,m.set_index('pair_id').loc[ids].neural_counterfactual_effect)
                        for _,m in null.groupby('method')]
                sensitivity.append(dict(theory=theory,split=split,endpoint=endpoint,total_pairs=len(frozen),
                    retained_pairs=len(ids),frozen_recovery=score,random_max=max(scores),
                    p_plus_one=(1+sum(x>=score for x in scores))/21))
        pd.DataFrame(sensitivity).to_csv(out/'norm_sensitivity.csv',index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('Replayed',len(data),'interventions')


if __name__=='__main__':main()
