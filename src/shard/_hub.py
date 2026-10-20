from __future__ import annotations

from pathlib import Path

from ._runtime import METADATA_FILE, MODEL_FILE, ModelFiles, check_size, read_dir


def download(repo_id: str, revision: str | None) -> ModelFiles:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise ImportError(
            f"{repo_id} is not a bundled model or a local directory, so it is treated as a Hugging Face "
            f'repository id, which needs huggingface_hub: pip install "shardac[hub]"'
        ) from e
    source = repo_id if revision is None else f"{repo_id}@{revision}"
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings or []}
    for name in (MODEL_FILE, METADATA_FILE):
        if name not in sizes:
            raise FileNotFoundError(f"{source}: {name} not found in the repository")
        check_size(name, sizes[name] or 0, source)
    path = snapshot_download(repo_id, revision=info.sha, allow_patterns=[MODEL_FILE, METADATA_FILE])
    return read_dir(Path(path), source)
