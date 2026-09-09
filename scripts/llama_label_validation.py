"""Post-pilot label correction, prospectively evaluated on independent situations.

Exactly two formats, X/Y and A/B, same seed and semantic conditions. No prompt
content changes, no threshold changes, no further format search in this run.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

from cognitive_discovery.pipeline import generate_design, load_config
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.data.validation import validate_records, pilot_gate
from cognitive_discovery.participants.qwen import QwenParticipant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revision', required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    root = Path('artifacts/llama_label_validation_v1')
    root.mkdir(parents=True, exist_ok=True)
    config = load_config('configs/discovery_v1.yaml')
    config.update(model='meta-llama/Llama-3.1-8B-Instruct', model_revision=args.revision,
                  output_root=str(root), design_seed=86001, split_seed=86002)
    config['collection']['expand_history_prefixes'] = False
    design, _ = generate_design(config, conditions=490, seed=86001,
                                design_id='llama_label_validation_v1')
    protocol = dict(model=config['model'], revision=args.revision, seed=86001,
                    labels=[['X','Y'],['A','B']], observations_per_format=len(design),
                    gates=config['collection'],
                    rationale='Original X/Y pilot failed counterbalance on all seven tasks.',
                    selection='One prospectively chosen A/B correction; X/Y is matched control; report both.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.prepare_only:
        return
    participant = QwenParticipant.from_pretrained(config['model'], revision=args.revision)
    start = time.monotonic()
    results = {}
    for name, labels in [('AB', ('A','B')), ('XY', ('X','Y'))]:
        records = []
        updated = []
        for c in design:
            m = c.response_mapping
            mapping = dataclasses.replace(m,
                        continue_label=labels[m.labels.index(m.continue_label)],
                        disengage_label=labels[m.labels.index(m.disengage_label)])
            updated.append(dataclasses.replace(c, response_mapping=mapping))
        (root/f'{name}_conditions.jsonl').write_text(''.join(json.dumps(c.to_dict())+'\n' for c in updated))
        for i, pid in enumerate(sorted({c.paired_condition_id for c in updated})):
            records.extend(collect_conditions([c for c in updated if c.paired_condition_id == pid],
                                               participant, model_revision=args.revision))
            if i % 20 == 0:
                print(name, len(records), len(design), flush=True)
        frame = validate_records(records)
        frame.to_parquet(root/f'{name}.parquet', index=False)
        results[name] = {task: pilot_gate(group, config) for task, group in frame.groupby('task_family')}
        (root/'gates.json').write_text(json.dumps(results, indent=2))
        print('GATES', name, json.dumps(results[name]), flush=True)
    complete = dict(seconds=time.monotonic()-start,
                    approved_AB=all(g['approved'] for g in results['AB'].values()),
                    approved_XY=all(g['approved'] for g in results['XY'].values()),
                    observations=2*len(design))
    (root/'complete.json').write_text(json.dumps(complete, indent=2))
    print('COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
