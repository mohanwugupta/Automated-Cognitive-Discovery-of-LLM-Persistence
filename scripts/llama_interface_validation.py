"""Predeclared polarity-counterbalanced yes/no interfaces; untouched validation."""
import argparse,dataclasses,hashlib,json,time
from pathlib import Path
from cognitive_discovery.pipeline import generate_design,load_config
from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.data.validation import validate_records,pilot_gate

VARIANTS=('question','proposition','instruction')
class InterfaceParticipant:
    def __init__(self,participant,variant):
        self.participant=participant;self.variant=variant;self.prompts=[]
        self.model_id=participant.model_id;self.revision=participant.revision
    def binary_decision(self,messages,labels,*,positive_label):
        text=messages[-1]['content'];prefix,tail=text.split('Choose one:\n')
        lines=tail.splitlines(); choices={line.split(' = ')[0]:line.split(' = ')[1] for line in lines if ' = ' in line}
        negative=next(x for x in labels if x!=positive_label)
        # Mapping flip changes which semantic proposition Yes endorses.
        queried=choices['Yes'];alternative=choices['No']
        if self.variant=='question':
            end=f'Your two possible actions are to {queried}, or to {alternative}.\nWould you choose to {queried}? Answer only Yes or No.'
        elif self.variant=='proposition':
            end=f'Evaluate this proposed decision: "I choose to {queried} instead of {alternative}."\nDoes this describe your decision? Answer only Yes or No.'
        else:
            end=f'Decide between these actions: {queried}; {alternative}.\nAnswer Yes if you choose to {queried}. Answer No if you choose to {alternative}.\nRespond with only Yes or No.'
        actual=[{'role':'user','content':prefix+end}]
        payload='\n'.join(f"{m['role']}:{m['content']}" for m in actual)
        self.prompts.append(dict(messages=actual,prompt_hash=hashlib.sha256(payload.encode()).hexdigest()))
        return self.participant.binary_decision(actual,('Yes','No'),positive_label=positive_label)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    out=Path('artifacts/llama_interface_v2');out.mkdir(parents=True,exist_ok=True)
    config=load_config('configs/discovery_v1.yaml');config['collection']['expand_history_prefixes']=False
    designs={}
    for phase,n,seed in [('calibration',210,91001),('validation',490,91002)]:
        raw,_=generate_design(config,output=out/phase,conditions=n,seed=seed,design_id='llama_interface_v2_'+phase)
        designs[phase]=[dataclasses.replace(c,response_mapping=dataclasses.replace(c.response_mapping,continue_label='Yes' if c.response_mapping.continue_label=='X' else 'No',disengage_label='No' if c.response_mapping.continue_label=='X' else 'Yes')) for c in raw]
        (out/f'{phase}_conditions.jsonl').write_text(''.join(json.dumps(c.to_dict())+'\n' for c in designs[phase]))
    protocol=dict(model='meta-llama/Llama-3.1-8B-Instruct',revision=a.revision,variants=VARIANTS,calibration_seed=91001,validation_seed=91002,gates=config['collection'],selection='All seven tasks must pass all inherited gates. Minimize worst-task polarity gap; lexical tie break. One independent validation attempt, no reselection or threshold changes.',estimand='Continuation probability from Yes/No logits under paired complementary action propositions; polarity gap replaces arbitrary-label mapping gap. Not identical to historical X/Y interface.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if a.prepare_only:return
    start=time.monotonic();participant=QwenParticipant.from_pretrained(protocol['model'],revision=a.revision)
    def collect(phase,variant):
        wrapper=InterfaceParticipant(participant,variant);rows=[];prompts=[]
        for i,pid in enumerate(sorted({c.paired_condition_id for c in designs[phase]})):
            wrapper.prompts=[];obs=collect_conditions([c for c in designs[phase] if c.paired_condition_id==pid],wrapper,model_revision=a.revision)
            assert len(obs)==len(wrapper.prompts)==2
            for row,prompt in zip(obs,wrapper.prompts):
                rows.append(dataclasses.replace(row,prompt_hash=prompt['prompt_hash']));prompts.append(dict(condition_id=row.condition_id,**prompt))
            if i%35==0:print('PROGRESS',phase,variant,len(rows),flush=True)
        frame=validate_records(rows);frame.to_parquet(out/f'{phase}_{variant}.parquet',index=False)
        (out/f'{phase}_{variant}_prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prompts))
        return {t:pilot_gate(g,config) for t,g in frame.groupby('task_family')}
    gates={}
    for v in VARIANTS:
        gates[v]=collect('calibration',v);(out/'calibration_gates.json').write_text(json.dumps(gates,indent=2));print('GATES',v,json.dumps(gates[v]),flush=True)
    candidates=[v for v in VARIANTS if all(g['approved'] for g in gates[v].values())]
    chosen=min(candidates,key=lambda v:(max(g['mean_absolute_mapping_gap'] for g in gates[v].values()),v)) if candidates else None
    result=dict(chosen=chosen,calibration_observations=3*len(designs['calibration']),validation_observations=0,validated=False)
    (out/'selection.json').write_text(json.dumps(result,indent=2))
    if chosen:
        vg=collect('validation',chosen);(out/'validation_gates.json').write_text(json.dumps(vg,indent=2));result.update(validation_observations=len(designs['validation']),validated=all(g['approved'] for g in vg.values()))
    result['seconds']=time.monotonic()-start;(out/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
