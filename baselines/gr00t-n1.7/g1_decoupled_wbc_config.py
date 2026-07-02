"""GR00T N1.7 modality config for SIMPLE G1 decoupled-WBC actions."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
]

ACTION_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
    "torso_vx",
    "torso_vy",
    "torso_vyaw",
    "target_yaw",
]

# The downloaded GR00T-N1.7-3B checkpoint is configured for a 40-step maximum
# horizon. We keep 40 as the default, but allow the training launcher to
# override it (e.g. retraining at 16 steps for a shorter control horizon).
ACTION_HORIZON = int(os.environ.get("GR00T_ACTION_HORIZON", "40"))
if ACTION_HORIZON <= 0:
    raise ValueError(f"GR00T_ACTION_HORIZON must be positive, got {ACTION_HORIZON}")

g1_decoupled_wbc_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["rs_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# This is a custom post-training embodiment for N1.7. The modality config is
# saved into the resulting checkpoint and reused for inference.
register_modality_config(
    g1_decoupled_wbc_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
