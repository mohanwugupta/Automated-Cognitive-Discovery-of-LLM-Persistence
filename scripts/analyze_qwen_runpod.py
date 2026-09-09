"""Analyze completed fixed-controller RunPod experiments without model inference."""
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from cognitive_discovery.causal_mechanistic.das import load_alignment
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics

root=Path('artifacts/qwen_runpod_v1')
# A clean checkout can reconstruct analysis inputs from the compact committed exports.
if not (root/'complete.json').exists():
    import shutil
    generated=Path('paper/generated')
    assert (generated/'runpod_complete.json').exists(), 'Wait for every predeclared method.'
    root.mkdir(parents=True,exist_ok=True)
    for name in ['metrics.csv','complete.json']:
        shutil.copy2(generated/('runpod_'+name),root/name)
    exported_pairs=pd.read_csv(generated/'runpod_pairs.csv')
    exported_pairs.drop(columns='pair_index').to_parquet(root/'pairs.parquet',index=False)
    exported=pd.read_csv(generated/'runpod_interventions.csv').merge(exported_pairs,on='pair_index',validate='many_to_one')
    exported['neural_counterfactual_effect']=exported.intervened_persistence_logit-exported.base_persistence_logit
    (root/'evaluations').mkdir(exist_ok=True)
    for name,group in exported.groupby('intervention_type'):
        group.to_parquet(root/'evaluations'/f'{name}.parquet',index=False)

metrics=pd.read_csv(root/'metrics.csv')
pairs=pd.read_parquet(root/'pairs.parquet')
angles=[]
paths=sorted((root/'alignments').glob('cognitive_DAS_*.safetensors'))
for a,b in itertools.combinations(paths,2):
    degrees=np.degrees(subspace_angles(load_alignment(a),load_alignment(b)))
    angles.append(dict(left=a.stem,right=b.stem,min_angle=float(degrees.min()),max_angle=float(degrees.max())))
if angles: pd.DataFrame(angles).to_csv(root/'principal_angles.csv',index=False)
frame=pd.read_parquet(root/'evaluations/frozen_DAS.parquet').drop(columns='contrast_id',errors='ignore').merge(pairs[['pair_id','contrast_id']],on='pair_id',validate='one_to_one')
fresh=frame[frame.evaluation_set.eq('fresh_signed_history')].copy().reset_index(drop=True)
fresh['pattern']=fresh.contrast_id.str.extract(r'_(\d+)$')[0].astype(int)
derangements=[p for p in itertools.permutations(range(4)) if all(i!=v for i,v in enumerate(p))]
rng=np.random.default_rng(76002); rows=[]
for split,g0 in fresh.groupby('pair_split'):
    g=g0.reset_index(drop=True)
    original=g.predicted_counterfactual_effect.to_numpy()
    observed=g.neural_counterfactual_effect.to_numpy()
    true=counterfactual_metrics(original,observed)
    null=[]
    for repeat in range(1000):
        shuffled=original.copy()
        for task,t in g.groupby('task_family'):
            perm=derangements[int(rng.integers(len(derangements)))]
            for _,mapping in t.groupby('response_mapping'):
                indexed=mapping.set_index('pattern').predicted_counterfactual_effect
                for index,row in mapping.iterrows():shuffled[index]=indexed.loc[perm[row.pattern]]
        assert np.all(np.abs(shuffled-original)>1e-8)
        score=counterfactual_metrics(shuffled,observed)['global_cfr'];null.append(score)
        rows.append(dict(split=split,repeat=repeat,shuffled_global_cfr=score,target_changed_fraction=1.0))
    true['empirical_p']=(1+sum(v>=true['global_cfr'] for v in null))/(len(null)+1)
    true['null_global_cfr']=float(np.mean(null));true['n']=len(g)
    (root/f'shuffle_{split}.json').write_text(json.dumps(true,indent=2))
pd.DataFrame(rows).to_csv(root/'shuffle_null.csv',index=False)
# Conditional uncertainty resamples semantic contrasts and keeps both mappings together.
rng=np.random.default_rng(76003);cis=[]
for split,g in frame[frame.evaluation_set.eq('original_pairs')].groupby('pair_split'):
    clusters=[v for _,v in g.groupby('contrast_id')];est=[]
    for _ in range(2000):
        boot=pd.concat([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
        est.append(counterfactual_metrics(boot.predicted_counterfactual_effect,boot.neural_counterfactual_effect)['global_cfr'])
    cis.append(dict(split=split,n=len(g),contrasts=len(clusters),ci_low=float(np.quantile(est,.025)),ci_high=float(np.quantile(est,.975))))
pd.DataFrame(cis).to_csv(root/'frozen_bootstrap.csv',index=False)
random_p=[]
for (split,dataset),g in metrics.groupby(['split','dataset']):
    frozen=float(g[g.method.eq('frozen_DAS')].global_cfr.iloc[0]);random=g[g.method.str.startswith('random_')].global_cfr
    random_p.append(dict(split=split,dataset=dataset,frozen_cfr=frozen,random_mean=float(random.mean()),random_sd=float(random.std()),empirical_p=float((1+(random>=frozen).sum())/(len(random)+1))))
pd.DataFrame(random_p).to_csv(root/'random_comparison.csv',index=False)
selected=metrics[~metrics.method.str.startswith('random_')]
print(selected[['method','split','dataset','global_cfr','correlation']].to_string(index=False))
print(pd.DataFrame(random_p).to_string(index=False))
# Compare every new seed with the original selected controller as well.
original_path=Path('artifacts/causal_mech_v1/representations/alignments/outcome_history__dual_history__L28__rank2__shared__all_tasks.safetensors')
for path in paths:
    degrees=np.degrees(subspace_angles(load_alignment(path),load_alignment(original_path)))
    angles.append(dict(left=path.stem,right='frozen_DAS',min_angle=float(degrees.min()),max_angle=float(degrees.max())))
if angles: pd.DataFrame(angles).to_csv(root/'principal_angles.csv',index=False)
if not angles:
    import shutil
    shutil.copy2(Path('paper/generated/runpod_principal_angles.csv'),root/'principal_angles.csv')
all_results=pd.concat([pd.read_parquet(f) for f in sorted((root/'evaluations').glob('*.parquet'))],ignore_index=True)
assert len(all_results)==31*120
assert all_results[['predicted_counterfactual_effect','neural_counterfactual_effect','intervention_norm']].notna().all().all()
assert not all_results.duplicated(['intervention_type','pair_id']).any()
for (method,split,dataset),group in all_results.groupby(['intervention_type','pair_split','evaluation_set']):
    recomputed=counterfactual_metrics(group.predicted_counterfactual_effect,group.neural_counterfactual_effect)['global_cfr']
    stored=metrics[(metrics.method==method)&(metrics.split==split)&(metrics.dataset==dataset)].global_cfr.iloc[0]
    assert np.isclose(recomputed,stored,rtol=1e-10,atol=1e-10), (method,split,dataset)

reference=frame.set_index('pair_id').intervention_norm
norm_audit=[]
for method,g in all_results.groupby('intervention_type'):
    if method.startswith('cognitive_DAS') or method=='frozen_DAS':continue
    target=reference.loc[g.pair_id].to_numpy();actual=g.intervention_norm.to_numpy()
    norm_audit.append(dict(method=method,max_absolute_delta=float(np.max(np.abs(actual-target))),median_relative_delta=float(np.median(np.abs(actual-target)/(target+1e-8)))))
pd.DataFrame(norm_audit).to_csv(root/'norm_audit.csv',index=False)
all_results.to_csv(root/'raw_results.csv',index=False)
# Figure: original and fresh evaluations remain distinct; no seed selection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
method_order=['frozen_DAS','cognitive_DAS','direct_DAS_75001','ridge_history_rank2','mean_difference_rank2','pca_rank2','ridge_persistence_rank2','random']
labels=['Frozen DAS','Cognitive DAS (5 seeds)','Direct DAS','History ridge + PCA','Mean difference + PCA','PCA','Persistence ridge + PCA','Random (20)']
for ax,split,title in zip(axes,['mech_pair_test','mech_task_holdout'],['Pair-test tasks','Held-out tasks']):
    for j,dataset in enumerate(['original_pairs','fresh_signed_history']):
        means=[];stds=[]
        for method in method_order:
            part=metrics[(metrics.split==split)&(metrics.dataset==dataset)]
            if method in ['cognitive_DAS','random']:part=part[part.method.str.startswith(method)]
            else:part=part[part.method==method]
            means.append(part.global_cfr.mean());stds.append(part.global_cfr.std() if len(part)>1 else 0)
        ax.errorbar(means,np.arange(8)+(j-.5)*.19,xerr=stds,fmt='o',ms=4,capsize=2,label=['Original pairs','Fresh signed histories'][j])
    ax.axvline(0,color='.6',lw=.7);ax.set_yticks(range(8),labels if ax is axes[0] else ['']*8);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Global counterfactual recovery');ax.grid(axis='x',alpha=.15)
handles, legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,fontsize=8,loc='outside lower center',ncol=2)
fig.savefig(root/'recovery_comparison.pdf');fig.savefig(root/'recovery_comparison.png',dpi=180);plt.close(fig)
# Compact, lossless row export: join pair_index to pairs.csv for task metadata.
pair_index={pair:i for i,pair in enumerate(pairs.pair_id)}
export=all_results[['intervention_type','pair_id','base_persistence_logit','intervened_persistence_logit','predicted_counterfactual_effect','intervention_norm']].copy()
export.insert(1,'pair_index',export.pair_id.map(pair_index));export=export.drop(columns='pair_id')
export.to_csv(root/'interventions.csv',index=False)
pairs.reset_index(names='pair_index').to_csv(root/'pairs.csv',index=False)
