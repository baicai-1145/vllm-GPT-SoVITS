# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    get_connectors_config_for_stage,
    load_omni_transfer_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = BASE_DIR / "vllm_omni" / "model_executor" / "stage_configs"


def get_config_files():
    """Helper to find config files."""
    if not CONFIG_DIR.exists():
        return []

    return list(CONFIG_DIR.glob("qwen*.yaml"))


# Collect files at module level for parametrization
config_files = get_config_files()


@pytest.mark.skipif(len(config_files) == 0, reason="No config files found or directory missing")
@pytest.mark.parametrize("yaml_file", config_files, ids=lambda p: p.name)
def test_load_qwen_yaml_configs(yaml_file):
    """
    Scan and test loading of all qwen*.yaml config files.
    This ensures that existing stage configs are compatible with the OmniConnector system.
    """
    print(f"Testing config load: {yaml_file.name}")
    try:
        # Attempt to load the config
        # default_shm_threshold doesn't matter much for loading correctness, using default
        config = load_omni_transfer_config(yaml_file)

        assert config is not None, "Config should not be None"

        # Basic validation
        # Note: Some configs might not have 'runtime' or 'connectors' section if they rely on auto-shm
        # but the load function should succeed regardless.

        # If the config defines stages, we expect connectors to be populated (either explicit or auto SHM)
        # We can't strictly assert len(config.connectors) > 0 because a single stage pipeline might have 0 edges.

        print(f"  -> Successfully loaded. Connectors: {len(config.connectors)}")

    except Exception as e:
        pytest.fail(f"Failed to load config {yaml_file.name}: {e}")


def test_load_gpt_sovits_v2_yaml_config():
    """GPT-SoVITS v2 stage config should expose its shared-memory edge via runtime.connectors."""
    yaml_file = CONFIG_DIR / "gpt_sovits_v2.yaml"
    assert yaml_file.exists(), f"Config not found: {yaml_file}"

    config = load_omni_transfer_config(yaml_file)

    assert config is not None
    assert ("0", "1") in config.connectors
    assert config.connectors[("0", "1")].name == "SharedMemoryConnector"

    stage_1_connectors = get_connectors_config_for_stage(config, stage_id=1)
    assert "from_stage_0" in stage_1_connectors
    assert stage_1_connectors["from_stage_0"]["spec"]["name"] == "SharedMemoryConnector"
