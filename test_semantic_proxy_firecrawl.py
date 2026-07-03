from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_proxy_module():
    path = Path("/Users/cleo/semantic-search-proxy.py")
    spec = importlib.util.spec_from_file_location("semantic_search_proxy_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_proxy_firecrawl_rotates_present_keys(monkeypatch, tmp_path, capsys) -> None:
    module = _load_proxy_module()
    monkeypatch.setattr(module, "COUNTER_FILE", str(tmp_path / "counter.json"))
    for idx in range(1, 7):
        monkeypatch.delenv(f"FIRECRAWL_API_KEY_{idx}", raising=False)
    monkeypatch.setenv("FIRECRAWL_API_KEY_1", "test-firecrawl-one")
    monkeypatch.setenv("FIRECRAWL_API_KEY_3", "test-firecrawl-three")

    first = module._get_current_firecrawl_key()
    second = module._force_advance_firecrawl()

    assert first["env_var"] == "FIRECRAWL_API_KEY_1"
    assert second["env_var"] == "FIRECRAWL_API_KEY_3"
    assert "firecrawl key rotation loaded 2/6 keys" in capsys.readouterr().out
