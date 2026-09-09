"""Build a local anonymous review-code package from an explicit allowlist.

Keeps the working repository intact. Omits git history, operational logs,
credentials, authorship notes, resource-account identifiers, and model weights.
"""
from pathlib import Path
import hashlib,json,zipfile
root=Path('.');out=Path('../../outputs/anonymous-review-code.zip');entries={}
for folder in ['src','configs']:
    for p in (root/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc'}:entries[str(p)]=p.read_bytes()
for name in ['pyproject.toml','scripts/analyze_llama_behavior.py','scripts/analyze_llama_mechanistic.py','scripts/analyze_qwen_confirmation.py','scripts/analyze_qwen_fresh_contexts.py','scripts/analyze_llama_pilots.py','scripts/analyze_llama_order.py','scripts/qwen_confirmation.py','scripts/qwen_fresh_contexts.py','scripts/qwen_fresh_natural_effects.py','scripts/llama_interface_validation.py','scripts/llama_behavior_replication.py','scripts/llama_mechanistic_replication.py']:
    p=Path(name)
    if p.exists():entries[name]=p.read_bytes()
for p in Path('paper/generated').rglob('*'):
    if not p.is_file() or p.suffix not in {'.csv','.gz','.json','.tex'}:continue
    if 'resource_audit' in p.name or 'access.json' in p.name or p.name=='audit.json':continue
    if p.stat().st_size>10_000_000:continue
    data=p.read_bytes()
    # Omit metadata with absolute user paths rather than silently changing data.
    if p.suffix!='.gz' and any(marker in data for marker in [b'/Users/',b'/home/',b'/scratch/',b'ssh-',b'RUNPOD_SECRET']):continue
    entries[str(p)]=data
for name in ['next-experiment-protocol.md','llama-replication-protocol.md','qwen-fresh-analysis-plan.md']:
    entries['paper/'+name]=Path('paper',name).read_bytes()
readme='''# Review code and compact results

This local review package contains analysis code, task implementations, declared protocols, and compact result exports. It omits operational identifiers, private credentials, model weights, and repository history. This is a candidate package requiring final author review; it has not been submitted.

From the package root, create a Python environment and install the project and parquet dependencies. For CPU-only result replay:

    python scripts/analyze_llama_behavior.py
    python scripts/analyze_llama_mechanistic.py
    PYTHONPATH=src python scripts/analyze_qwen_fresh_contexts.py

Once its completed exports are included, the independent Qwen confirmation can be replayed with:

    python scripts/analyze_qwen_confirmation.py

GPU reruns additionally require the pinned models, archived frozen behavioral packages and neural bases, and the separate raw task manifests. The compact package alone does not contain every original training artifact. Experiment runner paths document these dependencies; do not mistake CPU replay for a full model rerun.

A SHA-256 manifest identifies the exact files in this package. Original study numeric summaries are included for context; an independent arithmetic audit and full archives are distributed separately.
'''
entries['README.md']=readme.encode()
# Names of original authors may occur as legitimate bibliographic citations;
# none are needed in this code-only package. Operational markers must be absent.
markers=[b'runpod-llama',b'/Users/sandy',b'PRIVATE KEY',b'RUNPOD_SECRET_hf_token',b'7egy141v6r05l3',b'tk8jgwhvijy1rt']
findings=[name for name,data in entries.items() if any(m in data for m in markers)]
assert not findings,findings
entries['SHA256SUMS.json']=json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())},indent=2).encode()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(entries.items()):z.writestr(name,data)
print(json.dumps(dict(files=len(entries),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),operational_marker_scan='passed')))
