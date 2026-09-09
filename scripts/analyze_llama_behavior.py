"""Replay retained Llama behavioral scores from compact prediction exports."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score
root=Path('paper/generated/llama_behavior_v2');expected=pd.read_csv(root/'metrics.csv');rows=[]
for p in root.glob('*_predictions.csv'):
 phase,name=p.name.removesuffix('_predictions.csv').split('_',1);d=pd.read_csv(p)
 rows.append(dict(phase=phase,model=name,task='all',r2=r2_score(d.observed,d.prediction),n=len(d)))
 for task,g in d.groupby('task_family'):rows.append(dict(phase=phase,model=name,task=task,r2=r2_score(g.observed,g.prediction),n=len(g)))
actual=pd.DataFrame(rows);keys=['phase','model','task'];pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_like=True)
print('Replayed all Llama behavioral model scores.')
