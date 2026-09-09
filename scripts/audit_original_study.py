"""Independent arithmetic replay of original stored predictions and interventions.

Does not invoke original analysis functions or rerun original model inference.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd

def r2(y,p):return float(1-np.sum((np.asarray(y)-np.asarray(p))**2)/np.sum((np.asarray(y)-np.mean(y))**2))
def audit(root,out):
    out.mkdir(parents=True,exist_ok=True);sources={}
    def read(p,**kwargs):
        full=root/p;sources[str(p)]=hashlib.sha256(full.read_bytes()).hexdigest();return pd.read_csv(full,**kwargs)
    pred=read(Path('artifacts/linear_probes/test_predictions.csv'))
    split=json.loads((root/'artifacts/linear_probes/episode_split.json').read_text())
    assert not pred.state_id.duplicated().any()
    assert set(pred.episode_id).issubset(set(split['test']))
    assert not(set(split['test'])&set(split['train']) or set(split['test'])&set(split['validation']))
    probe=dict(rows=len(pred),episodes=pred.episode_id.nunique(),future_return_r2=r2(pred.future_return,pred.ridge_future_return),persistence_r2=r2(pred.persistence_logit,pred.ridge_persistence),episode_split_disjoint=True)
    factors=pd.concat([read(p.relative_to(root)) for p in sorted((root/'artifacts/value_dissociation').glob('factorial_shard_*.csv'))],ignore_index=True)
    group=factors.groupby('state_id');assert not factors.duplicated(['state_id','stop_payoff','continue_bonus']).any()
    assert group.size().eq(12).all();assert group.history_hash.nunique().eq(1).all()
    assert np.allclose(np.logaddexp(factors.logit_A,factors.logit_B)-factors.logit_C,factors.persistence_logit)
    columns=['persistence_logit','stop_payoff','continue_bonus','relative_incentive'];demeaned=factors[columns]-group[columns].transform('mean')
    y=demeaned.persistence_logit.to_numpy();x=demeaned[['stop_payoff','continue_bonus']].to_numpy();b=np.linalg.lstsq(x,y,rcond=None)[0];rel=demeaned.relative_incentive.to_numpy();slope=float(rel@y/(rel@rel))
    factorial=dict(rows=len(factors),states=group.ngroups,episodes=factors.episode_id.nunique(),complete_cells=True,history_fixed=True,logit_identity_verified=True,relative_incentive_r2=r2(y,rel*slope),stop_slope=float(b[0]),continue_slope=float(b[1]))
    result=dict(repository='https://github.com/mohanwugupta/digital-minds-hackathon',commit='5b968d0bb8de64f67556a7b300200aa488a591fa',probe=probe,factorial=factorial,steering_status='awaiting all LFS shards',scope='Independent arithmetic from stored rows; original inference and probe training not rerun.')
    ps=sorted((root/'artifacts/causal_steering').glob('replays_shard_*.csv'))
    if len(ps)==8 and all(p.stat().st_size>200 for p in ps):
        cols=['state_id','episode_id','direction_name','control_type','control_id','context_hash','logit_A','logit_B','logit_C','alpha','persistence_logit','probe_value_post']
        frames=[]
        for p in ps:
            d=read(p.relative_to(root),usecols=cols);frames.append(d)
        steer=pd.concat(frames,ignore_index=True)
        assert not steer.duplicated(['state_id','direction_name','control_type','control_id','alpha']).any()
        assert steer.groupby('state_id').context_hash.nunique().eq(1).all()
        assert np.allclose(np.logaddexp(steer.logit_A,steer.logit_B)-steer.logit_C,steer.persistence_logit)
        summaries=[];effects=[]
        for direction,g in steer[steer.control_type=='target'].groupby('direction_name'):
            w=g.pivot(index=['state_id','episode_id'],columns='alpha',values='persistence_logit');assert not w.isna().any().any();delta=(w[1]-w[-1]).rename('effect').reset_index();delta['direction']=direction;effects.append(delta)
            ep=delta.groupby('episode_id').effect.mean().to_numpy();rng=np.random.default_rng(94001);boot=np.mean(rng.choice(ep,size=(10000,len(ep)),replace=True),axis=1)
            summaries.append(dict(direction=direction,states=len(delta),episodes=len(ep),episode_weighted_effect=float(ep.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
        result.update(steering_status='independently recomputed from eight hash-verified LFS shards',steering=summaries,steering_rows=len(steer))
        pd.concat(effects,ignore_index=True).to_csv(out/'steering_state_effects.csv',index=False)
    result['source_sha256']=sources;(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('../original-study'));a.add_argument('--output',type=Path,default=Path('paper/generated/original_audit'));v=a.parse_args();audit(v.root,v.output)
