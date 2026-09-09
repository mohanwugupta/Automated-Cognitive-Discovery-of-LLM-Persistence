"""Prospective label-pair x semantic-option-order calibration diagnostic.

Calibration chooses among exactly four formats. A successful choice must pass
the inherited gates again on an independent validation design. No DAS fitting.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

from cognitive_discovery.pipeline import generate_design,load_config
from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.data.validation import validate_records,pilot_gate


class OrderedParticipant:
    def __init__(self, participant, reverse):
        self.participant=participant;self.reverse=reverse;self.prompts=[]
        self.model_id=participant.model_id;self.revision=participant.revision

    def binary_decision(self,messages,labels,*,positive_label):
        messages=[dict(m) for m in messages]
        if self.reverse:
            prefix, options=messages[-1]['content'].split('Choose one:\n')
            lines=options.splitlines()
            assert len(lines)>=4 and ' = ' in lines[0] and ' = ' in lines[1]
            lines[0],lines[1]=lines[1],lines[0]
            messages[-1]['content']=prefix+'Choose one:\n'+'\n'.join(lines)
        payload='\n'.join(f"{m['role']}:{m['content']}" for m in messages)
        self.prompts.append(dict(messages=messages,prompt_hash=hashlib.sha256(payload.encode()).hexdigest()))
        return self.participant.binary_decision(messages,labels,positive_label=positive_label)


def design_for(root,config,phase,count,seed):
    return generate_design(config,output=root/phase,conditions=count,seed=seed,
                design_id=f'llama_order_{phase}_v1')[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    root=Path('artifacts/llama_order_diagnostic_v1');root.mkdir(parents=True,exist_ok=True)
    config=load_config('configs/discovery_v1.yaml')
    config.update(model='meta-llama/Llama-3.1-8B-Instruct',model_revision=args.revision)
    config['collection']['expand_history_prefixes']=False
    calibration=design_for(root,config,'calibration',140,88001)
    validation=design_for(root,config,'validation',490,88002)
    variants=[('AB_continue_first',('A','B'),False),('AB_disengage_first',('A','B'),True),
              ('XY_continue_first',('X','Y'),False),('XY_disengage_first',('X','Y'),True)]
    protocol=dict(model=config['model'],revision=args.revision,calibration_seed=88001,validation_seed=88002,
        calibration_observations_per_format=len(calibration),validation_observations=len(validation),
        variants=[dict(name=n,labels=l,reverse_options=r) for n,l,r in variants],gates=config['collection'],
        selection='Only formats passing every gate in every task qualify; smallest maximum task mapping gap wins, lexical name tie-break. Evaluate one winner on untouched validation; no retry or reselection.',
        scope='diagnostic after earlier X/Y and A/B failures; no scientific controller fit',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if args.prepare_only:return
    start=time.monotonic()
    participant=QwenParticipant.from_pretrained(config['model'],revision=args.revision)
    def run(name,labels,reverse,design,phase):
        wrapper=OrderedParticipant(participant,reverse);records=[];prompts=[]
        mapped=[dataclasses.replace(c,response_mapping=dataclasses.replace(c.response_mapping,
            continue_label=labels[c.response_mapping.labels.index(c.response_mapping.continue_label)],
            disengage_label=labels[c.response_mapping.labels.index(c.response_mapping.disengage_label)])) for c in design]
        for i,pid in enumerate(sorted({c.paired_condition_id for c in mapped})):
            wrapper.prompts=[]
            observations=collect_conditions([c for c in mapped if c.paired_condition_id==pid],wrapper,
                                            model_revision=args.revision)
            assert len(observations)==len(wrapper.prompts)==2
            for observation,prompt in zip(observations,wrapper.prompts):
                records.append(dataclasses.replace(observation,prompt_hash=prompt['prompt_hash']))
                prompts.append(dict(condition_id=observation.condition_id,**prompt))
            if i%25==0:print(phase,name,len(records),len(design),flush=True)
        frame=validate_records(records)
        frame.to_parquet(root/f'{phase}_{name}.parquet',index=False)
        (root/f'{phase}_{name}_prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts))
        return {task:pilot_gate(g,config) for task,g in frame.groupby('task_family')}
    gates={}
    for name,labels,reverse in variants:
        gates[name]=run(name,labels,reverse,calibration,'calibration')
        (root/'calibration_gates.json').write_text(json.dumps(gates,indent=2))
        print('GATES',name,json.dumps(gates[name]),flush=True)
    candidates=[n for n,_,_ in variants if all(g['approved'] for g in gates[n].values())]
    chosen=min(candidates,key=lambda n:(max(g['mean_absolute_mapping_gap'] for g in gates[n].values()),n)) if candidates else None
    result=dict(chosen_format=chosen,calibration_observations=4*len(calibration),validation_observations=0)
    (root/'selection.json').write_text(json.dumps(result,indent=2))
    if chosen:
        name,labels,reverse=next(v for v in variants if v[0]==chosen)
        vg=run(name,labels,reverse,validation,'validation')
        (root/'validation_gates.json').write_text(json.dumps(vg,indent=2))
        result.update(validation_observations=len(validation),validation_passed=all(g['approved'] for g in vg.values()))
    result['seconds']=time.monotonic()-start
    (root/'complete.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
