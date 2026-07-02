"""GR00T N1.7 modality config for the G1 Sonic 46D -> 78D interface."""

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
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
]

ACTION_KEYS = [
    "motion_token",
    "left_hand_joints",
    "right_hand_joints",
]

ACTION_HORIZON = int(os.environ.get("GR00T_ACTION_HORIZON", "40"))
if ACTION_HORIZON <= 0:
    raise ValueError(f"GR00T_ACTION_HORIZON must be positive, got {ACTION_HORIZON}")

g1_sonic_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
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

register_modality_config(
    g1_sonic_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
