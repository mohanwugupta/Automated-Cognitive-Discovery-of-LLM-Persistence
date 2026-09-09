"""Conditional, prospectively selected Llama history-controller replication."""
import argparse,dataclasses,hashlib,json,pickle,time
from pathlib import Path
import numpy as np,pandas as pd
from qwen_fresh_contexts import prepare,dump
from llama_behavior_replication import REV,mapped
from cognitive_discovery.causal_mechanistic import pipeline as p
from cognitive_discovery.causal_mechanistic.counterfactuals import condition_feature_row
from cognitive_discovery.causal_mechanistic.das import DASAlignment,save_alignment
from cognitive_discovery.causal_mechanistic.metrics import counterfactual_metrics
from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner

def question_messages(messages):
    prefix,tail=messages[-1]['content'].split('Choose one:\n');choices={x.split(' = ')[0]:x.split(' = ')[1] for x in tail.splitlines() if ' = ' in x}
    return [{'role':'user','content':prefix+f"Your two possible actions are to {choices['Yes']}, or to {choices['No']}.\nWould you choose to {choices['Yes']}? Answer only Yes or No."}]
class QuestionRunner:
    def __init__(self,runner):self.runner=runner
    def forward(self,messages,labels,**kwargs):return self.runner.forward(question_messages(messages),labels,**kwargs)
    def differentiable_persistence_logit(self,messages,labels,**kwargs):return self.runner.differentiable_persistence_logit(question_messages(messages),labels,**kwargs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--design-only',action='store_true');a=ap.parse_args();out=Path('artifacts/llama_mechanistic_v2');out.mkdir(parents=True,exist_ok=True)
    protocol=dict(model='meta-llama/Llama-3.1-8B-Instruct',revision=REV,layers=[8,16,24,28],ranks=[2,8],epochs=3,learning_rate=.03,activation_penalty=.0001,search_seed=93104,random_seed=93105,random_count=99,design_seeds=[93101,93102,93103],selection='maximum global cognitive-target recovery on selection source tasks; lower rank then layer tie break',entry='requires independently successful llama_behavior_v2; no validity data used',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'protocol.json',protocol)
    if a.prepare_only:return
    behavior=Path('artifacts/llama_behavior_v2');status=json.loads((behavior/'complete.json').read_text())
    if not status['entry_passed']:dump(out/'blocked.json',{'reason':'behavioral entry criteria failed','behavior':status});return
    with (behavior/f"{status['chosen']}.pkl").open('rb') as f:theory=pickle.load(f)
    dump(out/'theory_freeze.json',dict(model=status['chosen'],sha256=hashlib.sha256((behavior/f"{status['chosen']}.pkl").read_bytes()).hexdigest()))
    allrecords=[];allpairs=[]
    for phase,seed in zip(['train','selection','test'],protocol['design_seeds']):
        records,pairs,_,_,_=prepare(out/phase,REV,seed=seed,patterns_override=[((-1,-1,-1),(1,-1,-1)),((1,1,1,1,1),(-1,-1,-1,1,1))])
        pairs=pairs[pairs.target_variable=='outcome_history'].copy()
        if phase!='test':pairs=pairs[pairs.pair_split!='mech_task_holdout']
        ids=set(pairs.base_condition_id)|set(pairs.source_condition_id);records=[r for r in records if r.condition.condition_id in ids]
        updated=[]
        for r in records:
            c=mapped(r.condition);c=dataclasses.replace(c,condition_id=phase+':'+c.condition_id,paired_condition_id=phase+':'+c.paired_condition_id);updated.append(dataclasses.replace(r,condition=c))
        for col in ['pair_id','base_condition_id','source_condition_id']:pairs[col]=phase+':'+pairs[col]
        pairs['phase']=phase;pairs.to_parquet(out/f'{phase}_pairs.parquet',index=False)
        allrecords.extend(updated);allpairs.append(pairs)
    pairs=pd.concat(allpairs,ignore_index=True);record_map={r.condition.condition_id:r for r in allrecords};assert len(record_map)==len(allrecords)
    (out/'conditions.jsonl').write_text(''.join(json.dumps(r.to_dict())+'\n' for r in allrecords))
    # No semantic-state overlap across neural fitting/selection/test phases.
    def key(r):return json.dumps(dict(task=r.condition.task_family,factors=r.condition.semantic_factors,history=dataclasses.asdict(r.condition.history)),sort_keys=True)
    sets={phase:{key(r) for r in allrecords if r.condition.condition_id.startswith(phase+':')} for phase in ['train','selection','test']}
    assert not(sets['train']&sets['selection'] or sets['train']&sets['test'] or sets['selection']&sets['test'])
    frame=pd.DataFrame([condition_feature_row(r.condition) for r in allrecords]);predicted=dict(zip(frame.condition_id,theory.predict(frame)))
    preds=pd.Series({r.pair_id:predicted[r.source_condition_id]-predicted[r.base_condition_id] for r in pairs.itertuples()})
    pairs['predicted_effect']=pairs.pair_id.map(preds);pairs.to_parquet(out/'pairs.parquet',index=False)
    if a.design_only:return
    import torch,transformers
    runner=MechanisticQwenRunner.from_pretrained(protocol['model'],revision=REV);wrapped=QuestionRunner(runner)
    for parameter in runner.model.parameters():parameter.requires_grad_(False)
    start=time.monotonic();cache={}
    # Cache only train/selection before winner freeze.
    for i,r in enumerate(allrecords):
        if r.condition.condition_id.startswith('test:'):continue
        cache[r.condition.condition_id]=p._forward(wrapped,r,capture_layers=protocol['layers'])
        if i%40==0:print('CACHE',i,flush=True)
    train=pairs[pairs.phase=='train'];selection=pairs[pairs.phase=='selection'];selection_metrics=[];candidates={}
    def baseline(ps):return pd.DataFrame([dict(pair_id=r.pair_id,base_persistence_logit=cache[r.base_condition_id].persistence_logit) for r in ps.itertuples()]).set_index('pair_id')
    for layer in protocol['layers']:
        states={cid:x.states[layer] for cid,x in cache.items()}
        for rank in protocol['ranks']:
            alignment=DASAlignment(len(next(iter(states.values()))),rank,seed=93104,device=next(runner.model.parameters()).device);optimizer=torch.optim.Adam(alignment.parameters(),lr=.03)
            for epoch in range(3):
                losses=[]
                for pair in train.sample(frac=1,random_state=93104+epoch).itertuples():
                    optimizer.zero_grad(set_to_none=True);captured={}
                    def editor(state):
                        edited=alignment.edit(state,states[pair.source_condition_id]);captured['penalty']=(edited-state).float().square().mean();return edited
                    rec=record_map[pair.base_condition_id];trial=p._trial(rec)
                    value=wrapped.differentiable_persistence_logit(list(trial.messages),rec.condition.response_mapping.labels,positive_label=rec.condition.response_mapping.continue_label,editors={layer:editor})
                    target=cache[pair.base_condition_id].persistence_logit+preds[pair.pair_id];loss=(value.float()-target).square()+.0001*captured['penalty'];loss.backward();assert torch.isfinite(alignment.parameter.grad).all();optimizer.step();losses.append(float(loss.detach()))
                print('TRAIN',layer,rank,epoch,float(np.mean(losses)),flush=True)
            basis=alignment.numpy_basis();name=f'L{layer}_r{rank}';candidates[name]=(layer,rank,basis);save_alignment(out/'candidates'/f'{name}.safetensors',basis,metadata={'training_pairs':train.pair_id.tolist()})
            rows=p._evaluate_alignment(wrapped,basis,selection,record_map,states,baseline(selection),preds,layer=layer,split_label='selection',intervention_type=name);df=pd.DataFrame(rows);df.to_parquet(out/f'selection_{name}.parquet',index=False);metrics=counterfactual_metrics(df.predicted_counterfactual_effect,df.neural_counterfactual_effect);selection_metrics.append(dict(name=name,layer=layer,rank=rank,**metrics));pd.DataFrame(selection_metrics).to_csv(out/'selection_metrics.csv',index=False)
    scores=pd.DataFrame(selection_metrics);winner=scores.sort_values(['global_cfr','rank','layer'],ascending=[False,True,True]).iloc[0];layer,rank,basis=candidates[winner['name']];dump(out/'selection.json',winner.to_dict());save_alignment(out/'selected.safetensors',basis,metadata=winner.to_dict());print('SELECTED',winner.to_dict(),flush=True)
    test=pairs[pairs.phase=='test']
    for r in allrecords:
        if r.condition.condition_id.startswith('test:'):cache[r.condition.condition_id]=p._forward(wrapped,r,capture_layers=(layer,))
    states={cid:x.states[layer] for cid,x in cache.items()};natural=test.copy();natural['natural_effect']=[cache[r.source_condition_id].persistence_logit-cache[r.base_condition_id].persistence_logit for r in test.itertuples()];natural.to_parquet(out/'natural.parquet',index=False)
    train_ids=sorted(set(train.base_condition_id)|set(train.source_condition_id));X=np.vstack([states[cid] for cid in train_ids]);y=np.array([cache[cid].persistence_logit for cid in train_ids]);_,_,vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
    from sklearn.linear_model import Ridge
    def complete(v):
        columns=[]
        for col in [v,*vt]:
            col=col.copy()
            for q in columns:col-=q*(q@col)
            if np.linalg.norm(col)>1e-8:columns.append(col/np.linalg.norm(col))
            if len(columns)==rank:break
        return np.column_stack(columns).astype(np.float32)
    rec=allrecords[0];trial=p._trial(rec);output=runner.choice_output_direction(question_messages(list(trial.messages)),rec.condition.response_mapping.labels,positive_label=rec.condition.response_mapping.continue_label)
    bases={'selected':basis,'ridge_persistence':complete(Ridge(alpha=1).fit(X,y).coef_),'output':complete(output),'pca':vt[:rank].T.astype(np.float32)};rng=np.random.default_rng(93105)
    for k in range(99):bases[f'random_{k:03d}']=np.linalg.qr(rng.normal(size=basis.shape),mode='reduced')[0].astype(np.float32)
    norms={r.pair_id:float(np.linalg.norm((states[r.source_condition_id]-states[r.base_condition_id])@basis)) for r in test.itertuples()}
    for name,b in bases.items():
        save_alignment(out/'bases'/f'{name}.safetensors',b,metadata={'method':name});rows=p._evaluate_alignment(wrapped,b,test,record_map,states,baseline(test),preds,layer=layer,split_label='manifest',intervention_type=name,matched_norms=None if name=='selected' else norms);df=pd.DataFrame(rows).drop(columns='pair_split').merge(test[['pair_id','pair_split','background','pattern']],on='pair_id',validate='one_to_one');df['method']=name;dest=out/'evaluations'/f'{name}.parquet';dest.parent.mkdir(exist_ok=True);df.to_parquet(dest,index=False);print('EVALUATED',name,len(df),flush=True)
    dump(out/'complete.json',dict(seconds=time.monotonic()-start,evaluation_rows=len(test)*len(bases),selected=winner.to_dict(),torch=torch.__version__,transformers=transformers.__version__,peak_vram_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
