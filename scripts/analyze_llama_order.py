"""Replay order-diagnostic acceptance gates and audit executed prompt hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cognitive_discovery.data.validation import pilot_gate

root=Path('artifacts/llama_order_diagnostic_v1')
out=Path('paper/generated/llama_order');out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((out/'protocol.json').read_text())
config={'collection':protocol['gates']}
summaries=[];allframes=[]
for variant in protocol['variants']:
    name=variant['name'];path=root/f'calibration_{name}.parquet'
    frame=pd.read_parquet(path) if path.exists() else pd.read_csv(out/f'{name}.csv')
    if path.exists():
        prompts=[json.loads(line) for line in (root/f'calibration_{name}_prompts.jsonl').read_text().splitlines()]
        byid={p['condition_id']:p for p in prompts}
        for row in frame.itertuples():
            prompt=byid[row.condition_id]
            actual='\n'.join(f"{m['role']}:{m['content']}" for m in prompt['messages'])
            assert hashlib.sha256(actual.encode()).hexdigest()==row.prompt_hash==prompt['prompt_hash']
        cols=['condition_id','paired_condition_id','episode_id','task_family','step','terminated',
              'p_continue','p_disengage','persistence_logit','top_token_is_action','p_action_mass_raw',
              'response_mapping','prompt_hash','model','model_revision']
        frame[cols].to_csv(out/f'{name}.csv',index=False)
    replay=pd.read_csv(out/f'{name}.csv')
    for task,group in replay.groupby('task_family'):
        gate=pilot_gate(group,config)
        summaries.append(dict(variant=name,task=task,n=len(group),
                     **{k:v for k,v in gate.items() if k!='checks'},
                     failed_checks=','.join(k for k,v in gate['checks'].items() if not v)))
    allframes.append(replay)
assert sum(len(f) for f in allframes)==1120
pd.DataFrame(summaries).to_csv(out/'task_gates.csv',index=False)
print(pd.DataFrame(summaries).groupby('variant').agg(passing_tasks=('approved','sum'),mean_p_continue=('mean_p_continue','mean')))
