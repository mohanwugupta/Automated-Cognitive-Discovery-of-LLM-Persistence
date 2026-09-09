"""Replay the held-out Llama neural replication, with clustered uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from analyze_qwen_confirmation import bootstrap,recovery

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('artifacts/llama_mechanistic_v2'));ap.add_argument('--output',type=Path,default=Path('paper/generated/llama_mechanistic_v2'));a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    raw=(a.root/'complete.json').exists();source=a.root if raw else out
    complete=json.loads((source/'complete.json').read_text())
    df=pd.concat([pd.read_parquet(p) for p in sorted((a.root/'evaluations').glob('*.parquet'))],ignore_index=True) if raw else pd.read_csv(out/'interventions.csv.gz')
    assert len(df)==complete['evaluation_rows']==11536
    natural=pd.read_parquet(a.root/'natural.parquet') if raw else pd.read_csv(out/'natural_effects.csv')
    if 'natural_effect' not in df:df=df.merge(natural[['pair_id','natural_effect']],on='pair_id',validate='many_to_one')
    assert not df.duplicated(['pair_id','method']).any()
    rows=[]
    for (split,method),g in df.groupby(['pair_split','method']):
        for endpoint in ['predicted_counterfactual_effect','natural_effect']:
            value=recovery(g[endpoint],g.neural_counterfactual_effect);lo,hi=(float('nan'),float('nan')) if method.startswith('random') else bootstrap(g,target=endpoint)
            rows.append(dict(split=split,method=method,endpoint=endpoint,n=len(g),recovery=value,low=lo,high=hi))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False);comps=[]
    # Report Holm correction separately per endpoint, plus across all four tests.
    for (split,endpoint),g in metrics.groupby(['split','endpoint']):
        selected=float(g[g.method=='selected'].recovery.iloc[0]);null=g[g.method.str.startswith('random')].recovery;assert len(null)==99
        comps.append(dict(split=split,endpoint=endpoint,selected=selected,random_mean=float(null.mean()),random_max=float(null.max()),p=float((1+(null>=selected).sum())/100)))
    comparisons=pd.DataFrame(comps)
    def holm(indices,column):
        order=comparisons.loc[indices].sort_values('p').index;adj=np.maximum.accumulate(np.minimum(1,comparisons.loc[order,'p'].to_numpy()*np.arange(len(order),0,-1)));comparisons.loc[order,column]=adj
    for _,g in comparisons.groupby('endpoint'):holm(g.index,'holm_within_endpoint')
    holm(comparisons.index,'holm_all_four');comparisons.to_csv(out/'random_comparisons.csv',index=False)
    controls=[]
    for split,g in df.groupby('pair_split'):
        selected=g[g.method=='selected']
        for method in ['ridge_persistence','output','pca']:
            c=selected.merge(g[g.method==method][['pair_id','neural_counterfactual_effect']],on='pair_id',suffixes=('','_control'),validate='one_to_one')
            for target in ['predicted_counterfactual_effect','natural_effect']:
                lo,hi=bootstrap(c,target=target,other='neural_counterfactual_effect_control');delta=recovery(c[target],c.neural_counterfactual_effect)-recovery(c[target],c.neural_counterfactual_effect_control)
                controls.append(dict(split=split,endpoint=target,control=method,recovery_difference=delta,low=lo,high=hi))
    pd.DataFrame(controls).to_csv(out/'control_differences.csv',index=False)
    norms=df.merge(df[df.method=='selected'][['pair_id','intervention_norm']].rename(columns={'intervention_norm':'selected_norm'}),on='pair_id',validate='many_to_one');norms['relative_error']=abs(norms.intervention_norm-norms.selected_norm)/norms.selected_norm.clip(lower=1e-12);norms.groupby(['pair_split','method']).relative_error.agg(['median','max']).reset_index().to_csv(out/'norm_audit.csv',index=False)
    # Post hoc precision sensitivity: common pairs, selected without outcome values.
    sensitivity=[]
    for split,g in norms.groupby('pair_split'):
        errors=g[g.method!='selected'].groupby('pair_id').relative_error.max()
        keep=errors[errors<=.05].index
        subset=g[g.pair_id.isin(keep)]
        for endpoint in ['natural_effect','predicted_counterfactual_effect']:
            scores={method:recovery(rows[endpoint],rows.neural_counterfactual_effect) for method,rows in subset.groupby('method')}
            null=[v for k,v in scores.items() if k.startswith('random')]
            sensitivity.append(dict(split=split,endpoint=endpoint,retained=len(keep),total=len(errors),selected=scores.get('selected',np.nan),random_max=max(null) if null else np.nan,p=(1+sum(v>=scores['selected'] for v in null))/100 if null else np.nan))
    pd.DataFrame(sensitivity).to_csv(out/'norm_sensitivity.csv',index=False)
    df.to_csv(out/'interventions.csv.gz',index=False,compression={'method':'gzip','mtime':0});natural.to_csv(out/'natural_effects.csv',index=False)
    for name in ['protocol.json','complete.json','selection.json','theory_freeze.json','selection_metrics.csv']:(out/name).write_bytes((source/name).read_bytes())
    print(comparisons.to_string(index=False));print(metrics[metrics.method=='selected'].to_string(index=False));print(pd.DataFrame(controls).to_string(index=False))
if __name__=='__main__':main()
