import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def metadata():
    import torch
    versions = {}
    for name in ('torch', 'triton', 'transformers', 'bitsandbytes', 'vllm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for folder in ('inference_engine', 'benchmarks')
               for path in sorted(Path(folder).glob('*.py'))}
    return dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                platform=platform.platform(), python=platform.python_version(),
                versions=versions, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(), source_sha256=sources,
                total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())


def save(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n')
