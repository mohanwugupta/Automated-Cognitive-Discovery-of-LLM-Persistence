"""Build a local anonymous review-code package from an explicit allowlist.

Keeps the working repository intact. Omits git history, operational logs,
credentials, authorship notes, resource-account identifiers, and model weights.
"""
from pathlib import Path
import gzip,hashlib,io,json,re,zipfile
import pandas as pd
root=Path('.');out=Path('../../outputs/anonymous-review-code.zip');entries={};sanitized=[]
for folder in ['src','configs']:
    for p in (root/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc'}:
            data=p.read_bytes()
            if folder=='configs' and p.suffix in {'.yaml','.yml'}:
                clean=re.sub(rb'^  (?:model_path_template|scratch_template):.*\n',b'',data,flags=re.MULTILINE)
                if clean!=data:
                    sanitized.append(dict(path=str(p),removed_keys=['cluster.model_path_template','cluster.scratch_template'],original_sha256=hashlib.sha256(data).hexdigest()))
                data=clean
            entries[str(p)]=data
for name in ['pyproject.toml','scripts/audit_computational_discrimination.py','scripts/analyze_llama_behavior.py','scripts/analyze_llama_mechanistic.py','scripts/analyze_qwen_confirmation.py','scripts/analyze_qwen_fresh_contexts.py','scripts/analyze_llama_pilots.py','scripts/analyze_llama_order.py','scripts/qwen_confirmation.py','scripts/qwen_fresh_contexts.py','scripts/qwen_fresh_natural_effects.py','scripts/llama_interface_validation.py','scripts/llama_behavior_replication.py','scripts/llama_mechanistic_replication.py']:
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
for name in ['computational-discrimination-audit.md','computational-discrimination-protocol.md','next-experiment-protocol.md','llama-replication-protocol.md','qwen-fresh-analysis-plan.md']:
    entries['paper/'+name]=Path('paper',name).read_bytes()
for name in ['design/contrast_manifest.parquet','frozen_das/interchange_results.parquet']:
    p = Path('artifacts/abstraction_discovery_v1') / name
    entries[str(p)] = p.read_bytes()
readme='''# Review code and compact results

This local review package contains analysis code, task implementations, declared protocols, and compact result exports. It omits operational identifiers, private credentials, model weights, and repository history. This is a candidate package requiring final author review; it has not been submitted.

From the package root, create a Python environment and install the project and parquet dependencies. For CPU-only result replay:

    python scripts/analyze_llama_behavior.py
    python scripts/analyze_llama_mechanistic.py
    PYTHONPATH=src python scripts/analyze_qwen_fresh_contexts.py

The completed independent Qwen confirmation can be replayed with:

    python scripts/analyze_qwen_confirmation.py

The descriptive computational-discrimination audit can be replayed with:

    python scripts/audit_computational_discrimination.py

Its two required original Parquet inputs are included. This is a post hoc audit, not a new confirmatory experiment.

GPU reruns additionally require the pinned models, archived frozen behavioral packages and neural bases, and the separate raw task manifests. The compact package alone does not contain every original training artifact. Experiment runner paths document these dependencies; do not mistake CPU replay for a full model rerun.

A SHA-256 manifest identifies the exact files in this package. Original study numeric summaries are included for context; an independent arithmetic audit and full archives are distributed separately.
'''
readme += '\nCluster model/scratch path templates are omitted from the review copy of nine configurations. Scientific settings are retained; original files are unchanged. PACKAGE_TRANSFORMS.json documents the exact files and original hashes. Supply local paths for GPU reruns.\n'
entries['README.md']=readme.encode()
entries['PACKAGE_TRANSFORMS.json']=json.dumps(sanitized,indent=2).encode()
# Names of original authors may occur as legitimate bibliographic citations;
# none are needed in this code-only package. Operational markers must be absent.
markers=[b'runpod-llama',b'/Users/sandy',b'PRIVATE KEY',b'RUNPOD_SECRET_hf_token',b'7egy141v6r05l3',b'tk8jgwhvijy1rt']
markers += [b'JORDANAT',b'/scratch/',b'/Users/',b'/home/',b'mohanwugupta',b'tanwisuth']
findings=[]
for name,data in entries.items():
    if name.endswith('.gz'): data=gzip.decompress(data)
    if name.endswith('.parquet'):
        data=pd.read_parquet(io.BytesIO(data)).to_csv(index=False).encode()
    if any(m.lower() in data.lower() for m in markers): findings.append(name)
assert not findings,findings
entries['SHA256SUMS.json']=json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())},indent=2).encode()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(entries.items()):z.writestr(name,data)
print(json.dumps(dict(files=len(entries),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),operational_marker_scan='passed')))
