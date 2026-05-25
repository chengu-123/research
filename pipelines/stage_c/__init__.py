"""Stage C — joint initialisation for Bootstrap B6.

Public API:
    StageCConfig, StageCInputs, JointInit, Psi (dataclasses)
    run_stage_c_joint_init(inputs, cfg, out_dir) -> JointInit

method.md section 6 B6 signature:
    psi_0, phi_0, anchors_object = stage_c_joint_init(
        z_final, M_attn_boot_64, O_init, is_carpet_mask, U_seed
    )

We deliver this via the `run_stage_c_joint_init` driver which returns a
typed `JointInit` dataclass. The legacy tuple form is available via
`JointInit.psi.pack_19()`, `JointInit.phi_0`, `JointInit.anchors_object`.
"""

from pipelines.stage_c.io_contract import StageCInputs, JointInit, Psi
from pipelines.stage_c.config import StageCConfig
from pipelines.stage_c.run_stage_c_init import run_stage_c_joint_init


__all__ = [
    "StageCConfig",
    "StageCInputs",
    "JointInit",
    "Psi",
    "run_stage_c_joint_init",
]
