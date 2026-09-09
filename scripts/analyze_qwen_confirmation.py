"""CPU replay of the prospectively specified Qwen confirmation endpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd

def recovery(target,effect):
    target=np.asarray(target);effect=np.asarray(effect);d=np.sum(target**2)
    return float(1-np.sum((effect-target)**2)/d) if d>1e-12 else float('nan')
def bootstrap(g, effect='neural_counterfactual_effect',target='natural_effect',other=None):
    g=g.copy();g['den']=g[target]**2;g['sse']=(g[effect]-g[target])**2
    if other:g['other_sse']=(g[other]-g[target])**2
    c=g.groupby(['task_family','background'])[['den','sse']+(['other_sse'] if other else [])].sum();rng=np.random.default_rng(92003);den=np.zeros(2000);sse=den.copy();other_sse=den.copy()
    for task in c.index.get_level_values(0).unique():
        x=c.loc[task].to_numpy();i=rng.integers(0,len(x),size=(2000,len(x)));s=x[i].sum(axis=1);den+=s[:,0];sse+=s[:,1]
        if other:other_sse+=s[:,2]
    value=(other_sse-sse)/den if other else 1-sse/den
    return np.quantile(value,[.025,.975]).tolist()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('artifacts/qwen_confirmation_v2'));ap.add_argument('--output',type=Path,default=Path('paper/generated/qwen_confirmation_v2'));a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    raw=(a.root/'complete.json').exists()
    source=a.root if raw else out
    complete=json.loads((source/'complete.json').read_text())
    if raw:
        frames=[pd.read_parquet(p) for p in sorted((a.root/'evaluations').glob('*.parquet'))];df=pd.concat(frames,ignore_index=True)
    else:df=pd.read_csv(out/'interventions.csv.gz')
    assert len(df)==complete['evaluation_rows']==23296
    natural=pd.concat([pd.read_parquet(a.root/f'natural_{v}.parquet') for v in ['original','rephrased']],ignore_index=True) if raw else pd.read_csv(out/'natural_effects.csv')
    if 'natural_effect' not in df:df=df.merge(natural[['pair_id','wording','natural_effect','original_prediction','global_prediction','task_prediction']],on=['pair_id','wording'],validate='many_to_one')
    assert not df.duplicated(['pair_id','wording','method']).any()
    rows=[];intervals=[];calibration=[]
    for (split,method),g in df.groupby(['pair_split','method']):
        for endpoint in ['natural_effect','original_prediction','global_prediction','task_prediction']:
            value=recovery(g[endpoint],g.neural_counterfactual_effect)
            rows.append(dict(split=split,method=method,endpoint=endpoint,n=len(g),recovery=value))
            if not method.startswith('random'):
                lo,hi=bootstrap(g,target=endpoint);intervals.append(dict(split=split,method=method,endpoint=endpoint,low=lo,high=hi))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False);pd.DataFrame(intervals).to_csv(out/'bootstrap.csv',index=False)
    comps=[]
    for split,g in metrics[metrics.endpoint=='natural_effect'].groupby('split'):
        value=float(g[g.method=='frozen'].recovery.iloc[0]);null=g[g.method.str.startswith('random')].recovery;assert len(null)==99
        comps.append(dict(split=split,frozen=value,random_mean=float(null.mean()),random_max=float(null.max()),p=float((1+(null>=value).sum())/100)))
    comparisons=pd.DataFrame(comps);order=np.argsort(comparisons.p.to_numpy());adjusted=np.maximum.accumulate(np.minimum(1,comparisons.p.to_numpy()[order]*np.arange(len(order),0,-1)));comparisons['holm_p']=np.nan;comparisons.loc[order,'holm_p']=adjusted;comparisons.to_csv(out/'random_comparisons.csv',index=False)
    controls=[]
    for split,g in df.groupby('pair_split'):
        frozen=g[g.method=='frozen']
        for method in ['ridge_persistence_rank2','direct_DAS_75001','pca_rank2','output_rank2']:
            c=frozen.merge(g[g.method==method][['pair_id','wording','neural_counterfactual_effect']],on=['pair_id','wording'],suffixes=('','_control'),validate='one_to_one')
            lo,hi=bootstrap(c,other='neural_counterfactual_effect_control');delta=recovery(c.natural_effect,c.neural_counterfactual_effect)-recovery(c.natural_effect,c.neural_counterfactual_effect_control)
            controls.append(dict(split=split,control=method,recovery_difference=delta,low=lo,high=hi))
    pd.DataFrame(controls).to_csv(out/'control_differences.csv',index=False)
    for (split,wording),g in natural.groupby(['pair_split','wording']):
        for col in ['original_prediction','global_prediction','task_prediction']:
            calibration.append(dict(split=split,wording=wording,predictor=col,recovery=recovery(g.natural_effect,g[col]),n=len(g)))
    pd.DataFrame(calibration).to_csv(out/'target_calibration.csv',index=False)
    granular=[]
    for (split,wording,method),g in df[~df.method.str.startswith('random')].groupby(['pair_split','wording','method']):granular.append(dict(split=split,wording=wording,method=method,n=len(g),natural_recovery=recovery(g.natural_effect,g.neural_counterfactual_effect)))
    pd.DataFrame(granular).to_csv(out/'wording_results.csv',index=False)
    norms=df.merge(df[df.method=='frozen'][['pair_id','wording','intervention_norm']].rename(columns={'intervention_norm':'frozen_norm'}),on=['pair_id','wording'],validate='many_to_one');norms['relative_error']=abs(norms.intervention_norm-norms.frozen_norm)/norms.frozen_norm.clip(lower=1e-12)
    norms.groupby(['pair_split','method']).relative_error.agg(['median','max']).reset_index().to_csv(out/'norm_audit.csv',index=False)
    # Post hoc finite-precision sensitivity; filter on norm error, never outcomes.
    sensitivity=[]
    for split,g in norms.groupby('pair_split'):
        errors=g[g.method!='frozen'].groupby(['pair_id','wording']).relative_error.max()
        keep=errors[errors<=.05].index
        subset=g.set_index(['pair_id','wording']).loc[keep].reset_index()
        scores={method:recovery(rows.natural_effect,rows.neural_counterfactual_effect) for method,rows in subset.groupby('method')}
        null=[v for k,v in scores.items() if k.startswith('random')]
        sensitivity.append(dict(split=split,retained=len(keep),total=len(errors),frozen=scores.get('frozen',np.nan),random_max=max(null) if null else np.nan,p=(1+sum(v>=scores['frozen'] for v in null))/100 if null else np.nan,**{m:scores.get(m,np.nan) for m in ['direct_DAS_75001','ridge_persistence_rank2','pca_rank2','output_rank2']}))
    pd.DataFrame(sensitivity).to_csv(out/'norm_sensitivity.csv',index=False)
    ablation=pd.concat([pd.read_parquet(a.root/f'ablation_{v}.parquet') for v in ['original','rephrased']],ignore_index=True) if raw else pd.read_csv(out/'ablation.csv')
    ab=[]
    for split,g in ablation.groupby('pair_split'):
        n=g.natural_effect.to_numpy();v=g.ablated_effect.to_numpy();ab.append(dict(split=split,projected_attenuation=float(1-n@v/(n@n)),residual_effect_rms=float(np.sqrt(np.mean(v*v))),natural_effect_rms=float(np.sqrt(np.mean(n*n))),n=len(g)))
    pd.DataFrame(ab).to_csv(out/'ablation_summary.csv',index=False)
    df.to_csv(out/'interventions.csv.gz',index=False,compression={'method':'gzip','mtime':0});natural.to_csv(out/'natural_effects.csv',index=False);ablation.to_csv(out/'ablation.csv',index=False)
    for name in ['protocol.json','complete.json','calibration.json']:(out/name).write_bytes((source/name).read_bytes())
    print(comparisons.to_string(index=False));print(pd.DataFrame(calibration).to_string(index=False));print(pd.DataFrame(controls).to_string(index=False))
if __name__=='__main__':main()
