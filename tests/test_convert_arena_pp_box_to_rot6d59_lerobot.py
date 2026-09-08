import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DATA = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPTS_DATA))
import convert_arena_pp_box_to_rot6d59_lerobot as converter  # noqa: E402


def test_sonic_qpos_is_reordered_to_rot6d59_body_order() -> None:
    source = np.arange(29, dtype=np.float32).reshape(1, 29)

    actual = converter.reorder_source_qpos_to_body29(
        source,
        source_family="sonic",
    )

    expected = source[:, converter.SONIC_TO_BODY29]
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, converter.arena.BODY29_ORDER.index("left_elbow_joint")] == 21
    assert actual[0, converter.arena.BODY29_ORDER.index("right_elbow_joint")] == 22


def test_twist2_qpos_already_uses_rot6d59_body_order() -> None:
    source = np.arange(58, dtype=np.float32).reshape(2, 29)

    actual = converter.reorder_source_qpos_to_body29(
        source,
        source_family="twist2",
    )

    np.testing.assert_array_equal(actual, source)
    assert not np.shares_memory(actual, source)


def test_unknown_qpos_source_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported source family"):
        converter.reorder_source_qpos_to_body29(
            np.zeros((1, 29), dtype=np.float32),
            source_family="unknown",
        )


def test_sonic_hands_are_reordered_from_provider_to_rot6d59_order() -> None:
    left = np.arange(7, dtype=np.float32).reshape(1, 7)
    right = (np.arange(7, dtype=np.float32) + 10).reshape(1, 7)

    actual = converter.hand14_from_source(
        left,
        right,
        source_family="sonic",
    )

    expected = np.concatenate(
        [left, right[:, converter.SONIC_RIGHT_TO_ROT59]], axis=1
    )
    np.testing.assert_array_equal(actual, expected)


def test_twist2_hands_keep_existing_arena_mapping() -> None:
    left = np.arange(7, dtype=np.float32).reshape(1, 7)
    right = (np.arange(7, dtype=np.float32) + 10).reshape(1, 7)

    actual = converter.hand14_from_source(
        left,
        right,
        source_family="twist2",
    )

    expected = converter.arena.hand14_from_arena(left, right)
    np.testing.assert_array_equal(actual, expected)


def test_metadata_documents_body_and_hand_reordering() -> None:
    info = converter.build_info(
        kind="pp_box",
        total_episodes=100,
        total_frames=123,
        total_videos=100,
        task="Pick up the box.",
        source_root=Path("/raw/pp_box"),
        branches=("sonic/yb",),
        mjcf_path=Path("/robot/g1.xml"),
        target_shift=1,
        hand_target_shift=0,
        skipped=[],
        video_codec="h264",
    )

    config = info["script_config"]
    assert config["schema_version"] == "arena_pp_box_rot6d59_v3"
    assert "reordered" in config["source_body29_orders"]["sonic"]
    assert "provider" in config["source_hand_orders"]["sonic"]


def test_open_door_twist2_metadata_is_not_labeled_as_sonic() -> None:
    info = converter.build_info(
        kind="open_door",
        total_episodes=100,
        total_frames=123,
        total_videos=100,
        task="Open the door.",
        source_root=Path("/raw/open_door"),
        branches=("twist2/zz",),
        mjcf_path=Path("/robot/g1.xml"),
        target_shift=1,
        hand_target_shift=0,
        skipped=[],
        video_codec="h264",
    )

    config = info["script_config"]
    assert config["source"] == "humanoid_arena_twist2_open_door_to_rot6d59"
    assert config["schema_version"] == "arena_open_door_twist2_rot6d59_v1"
    assert config["source_families"] == ["twist2"]
