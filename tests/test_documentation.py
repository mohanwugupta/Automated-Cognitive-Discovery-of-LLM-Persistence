from __future__ import annotations

from pathlib import Path

from cognitive_discovery.reproducibility.manifest import load_canonical_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_the_required_new_researcher_order():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = [
        "## 1. Scientific question",
        "## 2. Discovery methodology",
        "## 3. Canonical pipeline",
        "## 4. Main supported and bounded findings",
        "## 5. Reproduce manuscript numbers",
        "## 6. Rerun experiments",
        "## 7. Historical and supporting analyses",
        "## 8. Repository structure",
    ]
    offsets = [text.index(heading) for heading in headings]
    assert offsets == sorted(offsets)
    assert "python -m cognitive_discovery.reproduce claims" in text
    assert "python -m cognitive_discovery.run" in text


def test_required_stage_guides_and_reproduction_guide_exist():
    required = [
        ROOT / "docs/REPRODUCING_RESULTS.md",
        ROOT / "docs/stages/behavioral/README.md",
        ROOT / "docs/stages/mechanistic/README.md",
        ROOT / "docs/stages/generalization/README.md",
        ROOT / "docs/stages/replication/README.md",
        ROOT / "legacy/README.md",
    ]
    assert all(path.is_file() for path in required)


def test_every_claim_navigation_target_is_available_in_a_clean_clone():
    manifest = load_canonical_manifest(ROOT / "canonical_manifest.yaml")
    claims_text = (ROOT / "docs/CLAIMS.md").read_text(encoding="utf-8")
    for claim_id, claim in manifest["claims"].items():
        assert (ROOT / claim["code_path"]).is_file()
        if claim["config_path"] is not None:
            assert (ROOT / claim["config_path"]).is_file()
        assert all((ROOT / path).is_file() for path in claim["artifact_paths"])
        figure_path = claim["figure_ref"].split("#", maxsplit=1)[0]
        assert (ROOT / figure_path).is_file()
        assert f"python -m cognitive_discovery.reproduce claim {claim_id}" in claims_text
