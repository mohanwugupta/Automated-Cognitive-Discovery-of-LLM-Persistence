"""Secondary target-validity diagnostic; does not select or refit controllers."""
import json
from pathlib import Path
import pandas as pd
from cognitive_discovery.mechanistic.dataset.matched_conditions import load_mechanistic_manifest
from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
from cognitive_discovery.causal_mechanistic import pipeline as p

root=Path('artifacts/qwen_fresh_contexts_v1')
protocol=json.loads((root/'protocol.json').read_text())
pairs=pd.read_parquet(root/'pairs.parquet')
records={r.condition.condition_id:r for r in load_mechanistic_manifest(root/'conditions.jsonl')}
ids=sorted(set(pairs.source_condition_id))
(root/'natural_effects_protocol.json').write_text(json.dumps(dict(
    model=protocol['model'],revision=protocol['revision'],source_conditions=len(ids),
    role='secondary diagnostic added during intervention run; no fitting or selection',
    baseline='reuse independently cached base logits from the main run'),indent=2))
runner=MechanisticQwenRunner.from_pretrained(protocol['model'],revision=protocol['revision'])
values=[]
for i,cid in enumerate(ids):
    result=p._forward(runner,records[cid],capture_layers=())
    values.append(dict(condition_id=cid,persistence_logit=result.persistence_logit))
    if i%40==0:print('SOURCE',i,len(ids),flush=True)
sources=pd.DataFrame(values).set_index('condition_id').persistence_logit
baseline=pd.read_parquet(root/'baselines.parquet').base_persistence_logit
output=pairs[['pair_id','task_family','pair_split','target_variable']].copy()
output['natural_source_logit']=sources.loc[pairs.source_condition_id].to_numpy()
output['natural_base_logit']=baseline.loc[pairs.pair_id].to_numpy()
output['natural_effect']=output.natural_source_logit-output.natural_base_logit
output.to_parquet(root/'natural_effects.parquet',index=False)
print('COMPLETE',len(output),flush=True)
