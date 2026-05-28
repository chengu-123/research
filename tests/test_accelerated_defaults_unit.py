"""Static tests for accelerated Wan defaults.

These tests avoid loading Wan checkpoints. They guard the CLI/function
contracts that determine whether Stage A silently uses CPU offload.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.stage_a_wan import run_stage_a
from scripts import stagea


def test_run_stage_a_default_keeps_models_resident():
    sig = inspect.signature(run_stage_a)
    assert sig.parameters["offload_model"].default is False


def test_stagea_cli_default_keeps_models_resident(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stagea.py",
            "--image",
            "dummy.png",
            "--motion",
            "open the drawer",
            "--wan_ckpt",
            "dummy_ckpt",
            "--output_dir",
            "dummy_out",
        ],
    )
    args = stagea.parse_args()
    assert not hasattr(args, "offload_model")
    assert not hasattr(args, "no_offload_model")


def test_stagea_cli_has_no_t5_cpu_switch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stagea.py",
            "--image",
            "dummy.png",
            "--motion",
            "open the drawer",
            "--wan_ckpt",
            "dummy_ckpt",
            "--output_dir",
            "dummy_out",
        ],
    )
    args = stagea.parse_args()
    assert not hasattr(args, "t5_cpu")
    assert not hasattr(args, "fun_memory_mode")
