"""Independent behavioral entry test using the newly validated Llama interface."""
import argparse,dataclasses,json,hashlib,pickle,time
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score
from cognitive_discovery.pipeline import generate_design,load_config
from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.data.validation import validate_records,pilot_gate
from cognitive_discovery.hierarchy.random_effects import fit_hierarchical_model
from llama_interface_validation import InterfaceParticipant
REV='0e9e39f249a16976918f6564b8830bc894c89659'
def mapped(c):
    return dataclasses.replace(c,response_mapping=dataclasses.replace(c.response_mapping,continue_label='Yes' if c.response_mapping.continue_label=='X' else 'No',disengage_label='No' if c.response_mapping.continue_label=='X' else 'Yes'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    out=Path('artifacts/llama_behavior_v2');out.mkdir(parents=True,exist_ok=True)
    cfg=load_config('configs/discovery_v1.yaml');cfg['collection']['expand_history_prefixes']=False
    designs={};excluded=set()
    def key(c):return json.dumps(dict(task=c.task_family,factors=c.semantic_factors,history=dataclasses.asdict(c.history)),sort_keys=True)
    from cognitive_discovery.design.manifests import load_condition_manifest
    for prior in Path('artifacts/llama_interface_v2').glob('*_conditions.jsonl'):
        for c in load_condition_manifest(prior):excluded.add(key(c))
    for phase,n,seed in [('train',1400,93001),('selection',490,93002),('test',490,93003)]:
        cs,_=generate_design(cfg,output=out/phase,conditions=n*3,seed=seed,design_id='llama_behavior_'+phase)
        chosen=[]
        for task in sorted({c.task_family for c in cs}):
            count=0
            for c in cs:
                if c.task_family!=task or c.response_mapping.mapping_id!='continue_x' or key(c) in excluded:continue
                excluded.add(key(c));chosen.extend(x for x in cs if x.paired_condition_id==c.paired_condition_id);count+=1
                if count==n//7:break
            assert count==n//7,(phase,task,count)
        designs[phase]=[mapped(c) for c in chosen]
        (out/f'{phase}_conditions.jsonl').write_text(''.join(json.dumps(c.to_dict())+'\n' for c in designs[phase]))
    protocol=dict(revision=REV,interface='question',seeds=[93001,93002,93003],observations={k:len(v) for k,v in designs.items()},models=['immediate_state','choice_perseveration','outcome_history','dual_history'],variant='M3',alpha=1,gates=cfg['collection'],entry='history R2>=.6 and >= immediate+.01 on selection and untouched test; all final task measurement gates pass',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(Path('paper/llama-replication-protocol.md').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if a.prepare_only:return
    prior=Path('artifacts/llama_interface_v2/complete.json');assert json.loads(prior.read_text())['validated']
    participant=QwenParticipant.from_pretrained('meta-llama/Llama-3.1-8B-Instruct',revision=REV);start=time.monotonic();frames={};models={};metrics=[]
    for phase in ['train','selection','test']:
        wrapper=InterfaceParticipant(participant,'question');rows=[];prompts=[]
        for i,pid in enumerate(sorted({c.paired_condition_id for c in designs[phase]})):
            wrapper.prompts=[];obs=collect_conditions([c for c in designs[phase] if c.paired_condition_id==pid],wrapper,model_revision=REV)
            for row,prompt in zip(obs,wrapper.prompts):rows.append(dataclasses.replace(row,prompt_hash=prompt['prompt_hash']));prompts.append(dict(condition_id=row.condition_id,**prompt))
            if i%100==0:print('BEHAVIOR',phase,len(rows),flush=True)
        frame=validate_records(rows);frame.to_parquet(out/f'{phase}.parquet',index=False);frames[phase]=frame
        (out/f'{phase}_prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts))
        if phase=='train':
            for name in protocol['models']:
                fit=fit_hierarchical_model(frame,name,variant='M3',alphas=(1.,));models[name]=fit
                with (out/f'{name}.pkl').open('wb') as f:pickle.dump(fit,f)
                fit.task_parameters().to_csv(out/f'{name}_parameters.csv',index=False)
        for name,fit in models.items():
            prediction=fit.predict(frame);pd.DataFrame(dict(condition_id=frame.condition_id,task_family=frame.task_family,prediction=prediction,observed=frame.persistence_logit)).to_csv(out/f'{phase}_{name}_predictions.csv',index=False)
            metrics.append(dict(phase=phase,model=name,task='all',r2=r2_score(frame.persistence_logit,prediction),n=len(frame)))
            for task,g in frame.assign(prediction=prediction).groupby('task_family'):metrics.append(dict(phase=phase,model=name,task=task,r2=r2_score(g.persistence_logit,g.prediction),n=len(g)))
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        if phase=='selection':
            mm=pd.DataFrame(metrics);m=mm[(mm.phase=='selection')&(mm.task=='all')].set_index('model').r2
            chosen=max(['outcome_history','dual_history'],key=lambda x:(m[x],x));selection_pass=bool(m[chosen]>=.6 and m[chosen]>=m['immediate_state']+.01)
            (out/'selection.json').write_text(json.dumps(dict(chosen=chosen,passed=selection_pass,scores=m.to_dict()),indent=2))
            print('SELECTION',chosen,selection_pass,m.to_dict(),flush=True)
    mm=pd.DataFrame(metrics);m=mm[(mm.phase=='test')&(mm.task=='all')].set_index('model').r2
    gates={t:pilot_gate(g,cfg) for t,g in frames['test'].groupby('task_family')};(out/'test_gates.json').write_text(json.dumps(gates,indent=2))
    passed=bool(selection_pass and m[chosen]>=.6 and m[chosen]>=m['immediate_state']+.01 and all(x['approved'] for x in gates.values()))
    result=dict(chosen=chosen,entry_passed=passed,test_scores=m.to_dict(),observations=sum(map(len,frames.values())),seconds=time.monotonic()-start)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
