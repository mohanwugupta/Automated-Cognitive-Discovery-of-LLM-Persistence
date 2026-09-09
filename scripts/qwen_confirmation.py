"""Independent frozen-controller confirmation, with previously fitted target gains."""
import argparse,hashlib,json,time,dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
from qwen_fresh_contexts import prepare,dump
from cognitive_discovery.causal_mechanistic import pipeline as p
from cognitive_discovery.causal_mechanistic.das import load_alignment,save_alignment

REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
def transformed(messages,variant):
    if variant=='original':return [dict(m) for m in messages]
    text=messages[-1]['content']; context,choice=text.split('Choose one:\n')
    paragraphs=context.strip().split('\n\n')
    # Preserve every state/history statement verbatim; reverse only the two
    # context/factor paragraphs, and rephrase the response instruction.
    assert len(paragraphs)==4
    paragraphs[1],paragraphs[2]=paragraphs[2],paragraphs[1]
    choice=choice.replace('Respond with only X or Y.','Give exactly one answer: X or Y.')
    return [{'role':'user','content':'\n\n'.join(paragraphs)+'\n\nSelect your action:\n'+choice}]
class WordingRunner:
    def __init__(self,runner,variant):self.runner=runner;self.variant=variant
    def forward(self,messages,labels,**kwargs):return self.runner.forward(transformed(messages,self.variant),labels,**kwargs)

def calibrate():
    root=Path('paper/generated/qwen_fresh_contexts')
    old=pd.read_csv(root/'interventions.csv.gz');old=old[(old.theory=='dual_history')&(old.method=='frozen')]
    nat=pd.read_csv(root/'natural_effects.csv');d=old.merge(nat[['pair_id','natural_effect']],on='pair_id',validate='one_to_one')
    x=d.predicted_counterfactual_effect.to_numpy();y=d.natural_effect.to_numpy();gain=float(x@y/(x@x))
    task={}
    for t,g in d.groupby('task_family'):
        a=g.predicted_counterfactual_effect.to_numpy();b=g.natural_effect.to_numpy();task[t]=float((a@b+10*gain)/(a@a+10))
    return dict(global_gain=gain,task_gains=task,penalty=10,intercept=0,calibration_rows=len(d),source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['interventions.csv.gz','natural_effects.csv']})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    out=Path('artifacts/qwen_confirmation_v2');out.mkdir(parents=True,exist_ok=True)
    records,pairs,predictions,jobs,oldprotocol=prepare(out,REV,seed=92001,patterns_override=[((-1,-1,-1),(1,-1,-1)),((1,1,1,1,1),(-1,-1,-1,1,1))])
    pairs=pairs[pairs.target_variable=='outcome_history'].copy();pairs.to_parquet(out/'pairs.parquet',index=False)
    ids=set(pairs.base_condition_id)|set(pairs.source_condition_id);records=[r for r in records if r.condition.condition_id in ids]
    calibration=calibrate();dump(out/'calibration.json',calibration)
    job=next(j for j in jobs if j['theory']=='dual_history');basis=load_alignment(job['artifact'])
    # Check semantic novelty without depending on condition ids or random seed.
    def key(c):return json.dumps(dict(task=c.task_family,factors=c.semantic_factors,history=dataclasses.asdict(c.history)),sort_keys=True)
    from cognitive_discovery.mechanistic.dataset.matched_conditions import load_mechanistic_manifest
    previous=load_mechanistic_manifest('artifacts/qwen_fresh_contexts_v1/conditions.jsonl')
    overlap=set(key(r.condition) for r in records)&set(key(r.condition) for r in previous)
    assert not overlap, 'new design overlaps prior semantic states'
    protocol=dict(revision=REV,seed=92001,wordings=['original','rephrased'],pairs_per_wording=len(pairs),conditions_per_wording=len(records),random_count=99,random_seed=92002,primary='natural effect recovery; both task splits Holm corrected',calibration=calibration,controller=job,semantic_overlap_count=len(overlap),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(Path('paper/next-experiment-protocol.md').read_bytes()).hexdigest())
    dump(out/'protocol.json',protocol)
    pred=predictions[predictions.theory=='dual_history'].set_index('pair_id').predicted_counterfactual_effect
    prompts=[]
    for v in protocol['wordings']:
        for r in records:
            msgs=transformed(list(p._trial(r).messages),v);prompts.append(dict(condition_id=r.condition.condition_id,wording=v,messages=msgs,sha256=hashlib.sha256(json.dumps(msgs,sort_keys=True).encode()).hexdigest()))
    (out/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prompts))
    if a.prepare_only:return
    import torch,transformers
    from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
    runner=MechanisticQwenRunner.from_pretrained('Qwen/Qwen3.5-4B',revision=REV);start=time.monotonic()
    record_map={r.condition.condition_id:r for r in records}
    bases={'frozen':basis}
    for name in ['ridge_persistence_rank2','direct_DAS_75001','pca_rank2']:
        bases[name]=load_alignment(f'artifacts/qwen_runpod_v1/alignments/{name}.safetensors')
    trial=p._trial(records[0]);v=runner.choice_output_direction(list(trial.messages),records[0].condition.response_mapping.labels,positive_label=records[0].condition.response_mapping.continue_label)
    columns=[]
    for col in [v,*bases['pca_rank2'].T]:
        col=col.copy()
        for q in columns:col-=q*(q@col)
        if np.linalg.norm(col)>1e-8:columns.append(col/np.linalg.norm(col))
        if len(columns)==2:break
    bases['output_rank2']=np.column_stack(columns).astype(np.float32)
    rng=np.random.default_rng(92002)
    for k in range(99):bases[f'random_{k:03d}']=np.linalg.qr(rng.normal(size=basis.shape),mode='reduced')[0].astype(np.float32)
    for name,b in bases.items():save_alignment(out/'bases'/f'{name}.safetensors',b,metadata={'method':name})
    # Fixed original-training mean: no evaluation-state reference fitting.
    oldrecords=load_mechanistic_manifest('artifacts/causal_mech_v1/counterfactuals/mechanistic_conditions.jsonl')
    # The original train activations are collected explicitly below from original pair ids.
    original_pairs=pd.read_parquet('artifacts/causal_mech_v1/counterfactuals/pair_manifest.parquet')
    train=original_pairs[(original_pairs.target_variable=='outcome_history')&(original_pairs.pair_split=='mech_pair_train')]
    train_ids=set(train.base_condition_id)|set(train.source_condition_id)
    train_records=[r for r in oldrecords if r.condition.condition_id in train_ids]
    assert len(train_records)==len(train_ids)>0
    mean=np.mean([p._forward(runner,r,capture_layers=(28,)).states[28] for r in train_records],axis=0)
    np.save(out/'training_mean.npy',mean)
    for variant in protocol['wordings']:
        wrapped=WordingRunner(runner,variant);cache={}
        for i,(cid,r) in enumerate(record_map.items()):
            cache[cid]=p._forward(wrapped,r,capture_layers=(28,))
            if i%40==0:print('CACHE',variant,i,len(records),flush=True)
        states={cid:x.states[28] for cid,x in cache.items()}
        baseline=pd.DataFrame([dict(pair_id=r.pair_id,base_persistence_logit=cache[r.base_condition_id].persistence_logit) for r in pairs.itertuples()]).set_index('pair_id')
        natural=pairs.copy();natural['natural_effect']=[cache[r.source_condition_id].persistence_logit-cache[r.base_condition_id].persistence_logit for r in pairs.itertuples()]
        natural['original_prediction']=natural.pair_id.map(pred);natural['global_prediction']=natural.original_prediction*calibration['global_gain'];natural['task_prediction']=natural.original_prediction*natural.task_family.map(calibration['task_gains']);natural['wording']=variant
        natural.to_parquet(out/f'natural_{variant}.parquet',index=False)
        norms={r.pair_id:float(np.linalg.norm((states[r.source_condition_id]-states[r.base_condition_id])@basis)) for r in pairs.itertuples()}
        for name,b in bases.items():
            dest=out/'evaluations'/f'{variant}_{name}.parquet';dest.parent.mkdir(exist_ok=True)
            rows=p._evaluate_alignment(wrapped,b,pairs,record_map,states,baseline,pred,layer=28,split_label='manifest',intervention_type=name,matched_norms=None if name=='frozen' else norms)
            frame=pd.DataFrame(rows).drop(columns='pair_split').merge(pairs[['pair_id','pair_split','background','pattern']],on='pair_id',validate='one_to_one');frame['wording']=variant;frame['method']=name;frame.to_parquet(dest,index=False)
            print('EVALUATED',variant,name,len(frame),flush=True)
        # Mean replacement is a prespecified diagnostic with unscaled native norm.
        ablated={}
        for cid,r in record_map.items():
            def editor(state):
                u=torch.as_tensor(basis,device=state.device,dtype=torch.float32);m=torch.as_tensor(mean,device=state.device,dtype=torch.float32)
                return (state.float()+((m-state.float())@u)@u.T).to(state.dtype)
            ablated[cid]=p._forward(wrapped,r,layer=28,editor=editor).persistence_logit
        ablation=natural[['pair_id','task_family','pair_split','background','pattern','wording','natural_effect']].copy();ablation['ablated_effect']=[ablated[r.source_condition_id]-ablated[r.base_condition_id] for r in pairs.itertuples()];ablation.to_parquet(out/f'ablation_{variant}.parquet',index=False)
    dump(out/'complete.json',dict(seconds=time.monotonic()-start,evaluation_rows=len(pairs)*len(bases)*2,torch=torch.__version__,transformers=transformers.__version__,peak_vram_bytes=torch.cuda.max_memory_allocated()))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
