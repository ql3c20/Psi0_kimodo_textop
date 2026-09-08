import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DATA = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPTS_DATA))
import convert_arena_twist2_to_sonic_lerobot as converter  # noqa: E402


class ArrayData(dict[str, np.ndarray]):
    @property
    def files(self) -> list[str]:
        return list(self)


def episode_data(length: int = 4) -> ArrayData:
    root_quat = np.zeros((length, 4), dtype=np.float64)
    root_quat[:, 0] = 1.0
    return ArrayData(
        robot_qpos_before_decimation=np.zeros(
            (length, converter.BODY_DIM), dtype=np.float32
        ),
        robot_qvel_before_decimation=np.zeros(
            (length, converter.BODY_DIM), dtype=np.float32
        ),
        robot_root_orientation=root_quat,
        human_left_hand=np.arange(length * 7, dtype=np.float32).reshape(length, 7),
        human_right_hand=np.arange(length * 7, dtype=np.float32).reshape(length, 7)
        + 100.0,
        encoder_latent=np.arange(
            length * converter.TOKEN_DIM, dtype=np.float32
        ).reshape(length, converter.TOKEN_DIM),
    )


def test_recorded_encoder_latent_is_preserved_without_dropping_a_frame() -> None:
    data = episode_data()

    arrays = converter.load_episode_arrays(
        data,
        source_length=4,
        output_length=4,
        encoder=None,
        label="test",
        motion_token_source=converter.MOTION_TOKEN_SOURCE_RECORDED,
    )

    action = arrays[converter.ACTION_KEY]
    np.testing.assert_array_equal(action[:, :64], data["encoder_latent"])
    np.testing.assert_array_equal(
        action[:, 64:71],
        data["human_left_hand"][:, converter.SONIC_PROVIDER_TO_ACTION_HAND],
    )
    np.testing.assert_array_equal(
        action[:, 71:78],
        data["human_right_hand"][:, converter.SONIC_PROVIDER_TO_ACTION_HAND],
    )
    state = arrays[converter.STATE_KEY]
    np.testing.assert_array_equal(
        state[:, 22:29],
        data["human_left_hand"][:, converter.SONIC_PROVIDER_TO_STATE_HAND],
    )
    np.testing.assert_array_equal(
        state[:, 36:43],
        data["human_right_hand"][:, converter.SONIC_PROVIDER_TO_STATE_HAND],
    )
    assert state.shape == (4, 46)
    assert action.shape == (4, 78)


def test_recorded_sonic_qpos_is_reordered_for_native_state() -> None:
    data = episode_data(length=2)
    source_qpos = np.tile(
        np.arange(converter.BODY_DIM, dtype=np.float32), (2, 1)
    )
    data["robot_qpos_before_decimation"] = source_qpos
    data["robot_qvel_before_decimation"] = source_qpos + 100.0

    arrays = converter.load_episode_arrays(
        data,
        source_length=2,
        output_length=2,
        encoder=None,
        label="test",
        motion_token_source=converter.MOTION_TOKEN_SOURCE_RECORDED,
    )

    state = arrays[converter.STATE_KEY]
    actual_body = np.concatenate([state[:, :22], state[:, 29:36]], axis=1)
    expected_body = source_qpos[:, converter.pp_box.SONIC_TO_BODY29]
    np.testing.assert_array_equal(actual_body, expected_body)
    body_order = converter.pp_box.arena.BODY29_ORDER
    assert actual_body[0, body_order.index("left_elbow_joint")] == 21
    assert actual_body[0, body_order.index("right_elbow_joint")] == 22


@pytest.mark.parametrize("failure", ("missing", "short"))
def test_recorded_encoder_latent_rejects_missing_or_wrong_shape(failure: str) -> None:
    data = episode_data()
    if failure == "missing":
        del data["encoder_latent"]
        expected_error = KeyError
    else:
        data["encoder_latent"] = data["encoder_latent"][:-1]
        expected_error = ValueError

    with pytest.raises(expected_error, match="encoder_latent"):
        converter.load_episode_arrays(
            data,
            source_length=4,
            output_length=4,
            encoder=None,
            label="test",
            motion_token_source=converter.MOTION_TOKEN_SOURCE_RECORDED,
        )


def test_default_motion_token_source_remains_future_reencode() -> None:
    class FakeEncoder:
        def __init__(self) -> None:
            self.output_length: int | None = None

        def encode_episode(
            self,
            body_positions: np.ndarray,
            body_velocities: np.ndarray,
            root_rotations: np.ndarray,
            output_length: int,
            *,
            label: str,
        ) -> np.ndarray:
            del body_positions, body_velocities, root_rotations, label
            self.output_length = output_length
            return np.full((output_length, converter.TOKEN_DIM), 7.0, dtype=np.float32)

    encoder = FakeEncoder()
    arrays = converter.load_episode_arrays(
        episode_data(),
        source_length=4,
        output_length=3,
        encoder=encoder,  # type: ignore[arg-type]
        label="test",
    )

    assert encoder.output_length == 3
    np.testing.assert_array_equal(arrays[converter.ACTION_KEY][:, :64], 7.0)


def test_sonic_realized_future_reencode_uses_actual_qpos_in_encoder_order() -> None:
    class CaptureEncoder:
        def __init__(self) -> None:
            self.body_positions: np.ndarray | None = None
            self.body_velocities: np.ndarray | None = None

        def encode_episode(
            self,
            body_positions: np.ndarray,
            body_velocities: np.ndarray,
            root_rotations: np.ndarray,
            output_length: int,
            *,
            label: str,
        ) -> np.ndarray:
            del root_rotations, label
            self.body_positions = body_positions.copy()
            self.body_velocities = body_velocities.copy()
            return np.full(
                (output_length, converter.TOKEN_DIM), 11.0, dtype=np.float32
            )

    data = episode_data(length=4)
    source_qpos = np.arange(4 * converter.BODY_DIM, dtype=np.float32).reshape(4, 29)
    source_qvel = source_qpos + 1000.0
    data["robot_qpos_before_decimation"] = source_qpos
    data["robot_qvel_before_decimation"] = source_qvel
    encoder = CaptureEncoder()

    arrays = converter.load_episode_arrays(
        data,
        source_length=4,
        output_length=4,
        encoder=encoder,  # type: ignore[arg-type]
        label="sonic-realized",
        motion_token_source=converter.MOTION_TOKEN_SOURCE_FUTURE,
        source_family=converter.SOURCE_FAMILY_SONIC,
    )

    assert encoder.body_positions is not None
    assert encoder.body_velocities is not None
    np.testing.assert_array_equal(encoder.body_positions, source_qpos)
    np.testing.assert_array_equal(encoder.body_velocities, source_qvel)
    np.testing.assert_array_equal(arrays[converter.ACTION_KEY][:, :64], 11.0)
    np.testing.assert_array_equal(
        arrays[converter.ACTION_KEY][:, 64:71],
        data["human_left_hand"][:, converter.SONIC_PROVIDER_TO_ACTION_HAND],
    )
    state = arrays[converter.STATE_KEY]
    actual_body = np.concatenate([state[:, :22], state[:, 29:36]], axis=1)
    np.testing.assert_array_equal(
        actual_body,
        source_qpos[:, converter.pp_box.SONIC_TO_BODY29],
    )


def test_future_reencode_final_frame_policy() -> None:
    assert converter.output_length_for_episode(
        4, converter.MOTION_TOKEN_SOURCE_FUTURE, False
    ) == 3
    assert converter.output_length_for_episode(
        4, converter.MOTION_TOKEN_SOURCE_FUTURE, True
    ) == 4
    assert converter.output_length_for_episode(
        4, converter.MOTION_TOKEN_SOURCE_RECORDED, False
    ) == 4


def test_sonic_realized_metadata_documents_actual_future_motion() -> None:
    info = converter.build_info(
        kind="open_door",
        total_episodes=100,
        total_frames=83127,
        total_videos=100,
        task="Open the door.",
        source_root=Path("/raw/sonic/zz"),
        encoder_path=Path("/models/encoder.onnx"),
        skipped=[],
        reused_video_dataset=Path("/data/rot6d59"),
        motion_token_source=converter.MOTION_TOKEN_SOURCE_FUTURE,
        video_codec="mpeg4",
        source_family=converter.SOURCE_FAMILY_SONIC,
        keep_final_frame=True,
    )

    config = info["script_config"]
    assert config["schema_version"] == "arena_sonic_native_realized_v1"
    assert config["source_family"] == "sonic"
    assert config["motion_token_source"] == "future_reencode"
    assert config["drop_final_frame"] is False
    assert config["future_observation_indices"] == "t+1+5*k, k=0..9, clamp=end"
    assert "robot_qpos_before_decimation" in config["realized_body_source"]
    assert "reordered" in config["source_body29_order"]


def test_recorded_mode_metadata_documents_token_source() -> None:
    info = converter.build_info(
        kind="pp_box",
        total_episodes=100,
        total_frames=123,
        total_videos=100,
        task="Pick up the box.",
        source_root=Path("/raw/sonic/yb"),
        encoder_path=Path("/models/encoder.onnx"),
        skipped=[],
        reused_video_dataset=Path("/data/rot6d59"),
        motion_token_source=converter.MOTION_TOKEN_SOURCE_RECORDED,
        video_codec="h264",
    )

    config = info["script_config"]
    assert config["source"] == "humanoid_arena_sonic_to_native_sonic"
    assert config["schema_version"] == "arena_sonic_native_v2"
    assert config["drop_final_frame"] is False
    assert config["motion_token_source"] == "recorded_encoder_latent"
    assert config["future_observation_indices"] is None
    assert "reordered" in config["source_body29_order"]
    assert "reordered" in config["source_hand_order"]
