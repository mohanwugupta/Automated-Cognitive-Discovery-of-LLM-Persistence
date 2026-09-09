"""Prospective Llama behavioral feasibility pilot; no Qwen fitted targets.

Freezes the design before model access. Stops replication if any inherited
behavioral gate fails. Stores raw observations, prompts, and hook diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from cognitive_discovery.pipeline import generate_design, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--revision')
    args = ap.parse_args()
    root = Path('artifacts/llama_pilot_v1')
    root.mkdir(parents=True, exist_ok=True)
    config = load_config('configs/discovery_v1.yaml')
    config.update(model='meta-llama/Llama-3.1-8B-Instruct',
                  model_revision=args.revision, output_root=str(root),
                  design_seed=85001, split_seed=85002)
    config['collection']['expand_history_prefixes'] = False
    design, _ = generate_design(config, conditions=490, seed=85001,
                                design_id='llama_pilot_v1')
    protocol = dict(model=config['model'], revision=args.revision, seed=85001,
                    conditions=len(design), scope='behavioral feasibility, endpoint histories only',
                    gates=config['collection'],
                    progression='Every task must pass all inherited gates before replication.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.prepare_only:
        print(json.dumps(protocol), flush=True)
        return
    if not args.revision:
        raise ValueError('A verified immutable revision is required')
    import numpy as np
    import torch
    import transformers
    from cognitive_discovery.participants.qwen import QwenParticipant
    from cognitive_discovery.experiments.collection import collect_conditions
    from cognitive_discovery.experiments.registry import get_renderer
    from cognitive_discovery.data.validation import pilot_gate, validate_records
    from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
    from cognitive_discovery.causal_mechanistic.das import DASAlignment
    start = time.monotonic()
    participant = QwenParticipant.from_pretrained(config['model'], revision=args.revision)
    print('MODEL_LOADED', flush=True)
    observations = []
    pairs = sorted({c.paired_condition_id for c in design})
    for i, pair in enumerate(pairs):
        observations.extend(collect_conditions([c for c in design if c.paired_condition_id == pair],
                                               participant, model_revision=args.revision))
        if i % 10 == 0:
            print('BEHAVIOR', len(observations), len(design), flush=True)
            validate_records(observations).to_parquet(root/'partial.parquet', index=False)
    frame = validate_records(observations)
    frame.to_parquet(root/'behavior.parquet', index=False)
    prompts = [dict(condition_id=c.condition_id, messages=get_renderer(c.task_family).render(c).messages)
               for c in design]
    (root/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prompts))
    gates = {task: pilot_gate(group, config) for task, group in frame.groupby('task_family')}
    result = dict(approved=all(g['approved'] for g in gates.values()), tasks=gates)
    (root/'pilot_approval.json').write_text(json.dumps(result, indent=2))
    print('BEHAVIOR_GATES', json.dumps(result), flush=True)
    runner = MechanisticQwenRunner(participant)
    for param in runner.model.parameters():
        param.requires_grad_(False)
    base, source = design[0], next(c for c in design if c.task_family == design[0].task_family
                                 and c.response_mapping == design[0].response_mapping
                                 and c.paired_condition_id != design[0].paired_condition_id)
    layer = runner.layer_count // 2
    messages = list(get_renderer(base.task_family).render(base).messages)
    labels = base.response_mapping.labels
    positive = base.response_mapping.continue_label
    natural = runner.forward(messages, labels, positive_label=positive, capture_layers=[layer])
    identity = runner.forward(messages, labels, positive_label=positive, capture_layers=[],
                              editors={layer: lambda x: x})
    src = runner.forward(list(get_renderer(source.task_family).render(source).messages),
                         labels, positive_label=positive, capture_layers=[layer]).states[layer]
    align = DASAlignment(len(src), 2, seed=85003, device=next(runner.model.parameters()).device)
    value = runner.differentiable_persistence_logit(messages, labels, positive_label=positive,
                    editors={layer: lambda x: align.edit(x, src)})
    value.backward()
    hook = dict(layer=layer, identity_error=abs(identity.persistence_logit-natural.persistence_logit),
                gradient_finite=bool(torch.isfinite(align.parameter.grad).all()),
                gradient_norm=float(torch.linalg.vector_norm(align.parameter.grad)),
                intervention_effect=float(value.detach())-natural.persistence_logit,
                state_finite=bool(np.isfinite(src).all()))
    (root/'hook_check.json').write_text(json.dumps(hook, indent=2))
    complete = dict(behavioral_approved=result['approved'], hook_check=hook,
                    observations=len(frame), seconds=time.monotonic()-start,
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(), peak_vram_bytes=torch.cuda.max_memory_allocated())
    (root/'complete.json').write_text(json.dumps(complete, indent=2))
    print('COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
