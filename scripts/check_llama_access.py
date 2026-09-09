"""Read-only Hugging Face gated-weight access check; no weight downloads.

Use HF_TOKEN injected by RunPod Secrets or an existing Hugging Face login.
Outputs status and immutable model revision, never credentials or raw errors.
"""
import json
import sys

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    token = get_token()
    if not token:
        print(json.dumps({"model": MODEL, "status": "credential_missing"}))
        return 2
    try:
        info = HfApi(token=token).model_info(MODEL, files_metadata=False)
        weights = sorted(s.rfilename for s in info.siblings
                         if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename)
        if not weights:
            print(json.dumps({"model": MODEL, "status": "weight_file_missing"}))
            return 3
        # HEAD of an actual weight file establishes access; public model metadata does not.
        metadata = get_hf_file_metadata(hf_hub_url(MODEL, weights[0], revision=info.sha), token=token)
        print(json.dumps({"model": MODEL, "status": "gated_weight_access_verified",
                          "revision": info.sha, "weight_file": weights[0], "weight_bytes": metadata.size}))
        return 0
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(json.dumps({"model": MODEL, "status": "access_check_failed",
                          "http_status": status, "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
