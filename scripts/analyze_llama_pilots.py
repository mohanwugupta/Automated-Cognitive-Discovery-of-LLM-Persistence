"""Export and independently replay Llama pilot gates from compact observations."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from cognitive_discovery.data.validation import pilot_gate


def main():
    output = Path('paper/generated/llama_pilots')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, variants in [('llama_pilot_v1', [('original_XY','behavior')]),
                          ('llama_label_validation_v1', [('fresh_AB','AB'),('fresh_XY','XY')])]:
        root = Path('artifacts')/run
        protocol = json.loads((root/'protocol.json').read_text())
        config = {'collection':protocol['gates']}
        for name, filename in variants:
            frame = pd.read_parquet(root/f'{filename}.parquet')
            cols = ['condition_id','paired_condition_id','task_family','episode_id','step','terminated',
                    'p_continue','p_disengage','persistence_logit','top_token_is_action',
                    'p_action_mass_raw','response_mapping','model','model_revision']
            export = output/f'{name}_observations.csv'
            frame[cols].to_csv(export, index=False)
            replay = pd.read_csv(export)
            for task, group in replay.groupby('task_family'):
                gate = pilot_gate(group, config)
                original = pilot_gate(frame[frame.task_family.eq(task)], config)
                for k in ['mean_p_continue','mean_absolute_mapping_gap','persistence_logit_sd']:
                    assert np.isclose(gate[k], original[k], atol=1e-12)
                rows.append(dict(run=name, task=task, observations=len(group),
                                 **{k:v for k,v in gate.items() if k!='checks'},
                                 failed_checks=','.join(k for k,v in gate['checks'].items() if not v)))
        for file in ['protocol.json','complete.json','pilot_approval.json','hook_check.json','gates.json','access.json']:
            if (root/file).exists():
                shutil.copy2(root/file, output/f'{run}_{file}')
    pd.DataFrame(rows).to_csv(output/'task_gates.csv', index=False)
    print(pd.DataFrame(rows)[['run','task','mean_absolute_mapping_gap','approved']].to_string(index=False))


if __name__ == '__main__':
    main()
