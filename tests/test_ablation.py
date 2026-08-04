import importlib.util
from pathlib import Path


def _load_run_ablation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_dataset_path_falls_back_to_cuad_config():
    module = _load_run_ablation_module()
    config = {
        "evaluation": {},
        "cuad": {"eval_dataset_path": "data/cuad_eval.json"},
    }

    resolved = module._resolve_dataset_path(config, config_dir=Path("/tmp/project"))

    assert resolved == Path("/tmp/project/data/cuad_eval.json")
