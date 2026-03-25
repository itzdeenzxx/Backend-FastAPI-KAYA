import glob
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import interp1d

try:
    import tensorflow as tf

    def load_tflite(model_path: str):
        # Try SELECT_TF_OPS first (supports TensorFlow Flex ops like FlexTensorListReserve)
        resolvers = [
            tf.lite.experimental.OpResolverType.SELECT_TF_OPS,
            tf.lite.experimental.OpResolverType.AUTO,
            tf.lite.experimental.OpResolverType.BUILTIN_REF,
        ]

        for resolver in resolvers:
            try:
                return tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_op_resolver_type=resolver
                )
            except (ValueError, RuntimeError, Exception):
                continue

        # Final fallback: default interpreter without experimental resolver
        return tf.lite.Interpreter(model_path=model_path)

except ImportError:
    import tflite_runtime.interpreter as tflite

    def load_tflite(model_path: str):
        return tflite.Interpreter(model_path=model_path)


class PoseServiceError(Exception):
    def __init__(self, code: str, message: str, details=None, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code


@dataclass
class ModelBundle:
    pose_name: str
    cfg: dict
    lstm_interp: object
    cnn_interp: object
    lock: threading.Lock


class PoseScoringService:
    def __init__(self, models_root: str):
        self.models_root = models_root
        self._bundles: Dict[str, ModelBundle] = {}
        self._alias_to_pose: Dict[str, str] = {}
        self._load_all_models()

    @staticmethod
    def _normalize_pose_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _load_all_models(self) -> None:
        if not os.path.isdir(self.models_root):
            raise PoseServiceError(
                code="MODELS_DIR_NOT_FOUND",
                message="Models directory does not exist.",
                details={"models_root": self.models_root},
                status_code=500,
            )

        pose_dirs = [
            d
            for d in os.listdir(self.models_root)
            if os.path.isdir(os.path.join(self.models_root, d))
        ]

        for pose_name in sorted(pose_dirs):
            pose_dir = os.path.join(self.models_root, pose_name)
            bundle = self._load_bundle(pose_name, pose_dir)
            self._bundles[pose_name] = bundle

            normalized = self._normalize_pose_name(pose_name)
            self._alias_to_pose[normalized] = pose_name

            # Common aliases from frontend naming style.
            self._alias_to_pose[pose_name.lower()] = pose_name
            self._alias_to_pose[pose_name.replace("-", "").lower()] = pose_name

        if not self._bundles:
            raise PoseServiceError(
                code="NO_MODELS_FOUND",
                message="No pose model folders were found under models root.",
                details={"models_root": self.models_root},
                status_code=500,
            )

    def _load_bundle(self, pose_name: str, pose_dir: str) -> ModelBundle:
        lstm_candidates = sorted(glob.glob(os.path.join(pose_dir, "lstm_autoencoder*.tflite")))
        cnn_candidates = sorted(glob.glob(os.path.join(pose_dir, "cnn_autoencoder*.tflite")))
        cfg_candidates = sorted(glob.glob(os.path.join(pose_dir, "scoring_config*.json")))

        if not lstm_candidates or not cnn_candidates or not cfg_candidates:
            raise PoseServiceError(
                code="MODEL_FILES_MISSING",
                message="Model files are missing for one or more poses.",
                details={
                    "pose_name": pose_name,
                    "lstm_found": len(lstm_candidates),
                    "cnn_found": len(cnn_candidates),
                    "config_found": len(cfg_candidates),
                },
                status_code=500,
            )

        with open(cfg_candidates[0], "r", encoding="utf-8") as f:
            cfg = json.load(f)

        return ModelBundle(
            pose_name=pose_name,
            cfg=cfg,
            lstm_interp=load_tflite(lstm_candidates[0]),
            cnn_interp=load_tflite(cnn_candidates[0]),
            lock=threading.Lock(),
        )

    def list_poses(self) -> List[str]:
        return sorted(self._bundles.keys())

    def resolve_pose_name(self, pose_name: str) -> str:
        key = self._normalize_pose_name(pose_name)
        resolved = self._alias_to_pose.get(key)
        if not resolved:
            raise PoseServiceError(
                code="UNKNOWN_POSE",
                message="Pose not found. Use one of available poses.",
                details={"requested_pose": pose_name, "available_poses": self.list_poses()},
                status_code=404,
            )
        return resolved

    def score(self, pose_name: str, raw_frames: List[List[List[float]]]) -> Dict[str, float]:
        resolved_pose = self.resolve_pose_name(pose_name)
        bundle = self._bundles[resolved_pose]

        raw_np = self._to_numpy_frames(raw_frames)
        raw_clip = self._preprocess_raw_only(raw_np, bundle.cfg)
        if raw_clip is None:
            raise PoseServiceError(
                code="INSUFFICIENT_FRAMES",
                message="At least 2 frames are required after parsing.",
                details={"received_frames": int(raw_np.shape[0])},
                status_code=422,
            )

        move_gate, move_ratio = self._compute_movement_gate(raw_clip, bundle.cfg)
        clip = self._scale_raw_clip(raw_clip, bundle.cfg)

        with bundle.lock:
            lstm_pred = self._tflite_predict(bundle.lstm_interp, clip)
            cnn_pred = self._tflite_predict(bundle.cnn_interp, clip)

        lstm_error = float(np.mean(np.square(clip - lstm_pred)))
        cnn_error = float(np.mean(np.square(clip - cnn_pred)))

        lstm_threshold = float(bundle.cfg.get("lstm_threshold", 0.01))
        cnn_threshold = float(bundle.cfg.get("cnn_threshold", 0.01))

        if lstm_threshold <= 0 or cnn_threshold <= 0:
            raise PoseServiceError(
                code="INVALID_THRESHOLD",
                message="Model threshold in config must be greater than 0.",
                details={
                    "pose_name": resolved_pose,
                    "lstm_threshold": lstm_threshold,
                    "cnn_threshold": cnn_threshold,
                },
                status_code=500,
            )

        # Keep the same scoring style as pose_scoring_app.py
        lstm_score = float(np.exp(-lstm_error / (lstm_threshold * 10.0)) * 100.0)
        cnn_score = float(np.exp(-cnn_error / (cnn_threshold * 10.0)) * 100.0)

        lstm_final = lstm_score * move_gate
        cnn_final = cnn_score * move_gate
        avg_final = lstm_final * 0.3 + cnn_final * 0.7

        return {
            "lstm_score": round(lstm_final, 1),
            "cnn_score": round(cnn_final, 1),
            "avg_score": round(avg_final, 1),
            "lstm_error": round(lstm_error, 6),
            "cnn_error": round(cnn_error, 6),
            "move_gate": round(move_gate, 3),
            "move_ratio": round(move_ratio, 4),
        }

    @staticmethod
    def _to_numpy_frames(raw_frames: List[List[List[float]]]) -> np.ndarray:
        try:
            arr = np.asarray(raw_frames, dtype=np.float32)
        except Exception as exc:
            raise PoseServiceError(
                code="INVALID_INPUT_FORMAT",
                message="raw_frames must be numeric with shape (N, 33, 2).",
                details={"error": str(exc)},
                status_code=422,
            ) from exc

        if arr.ndim != 3 or arr.shape[1:] != (33, 2):
            raise PoseServiceError(
                code="INVALID_INPUT_SHAPE",
                message="raw_frames must have shape (N, 33, 2).",
                details={"received_shape": list(arr.shape)},
                status_code=422,
            )

        if arr.shape[0] < 2:
            raise PoseServiceError(
                code="INSUFFICIENT_FRAMES",
                message="Need at least 2 frames to interpolate and score.",
                details={"received_frames": int(arr.shape[0])},
                status_code=422,
            )

        if not np.isfinite(arr).all():
            raise PoseServiceError(
                code="NON_FINITE_VALUES",
                message="raw_frames contains NaN or infinite values.",
                status_code=422,
            )

        return arr

    @staticmethod
    def _body_joint_indices(cfg: dict) -> List[int]:
        if "body_joints" in cfg:
            return list(cfg["body_joints"])
        if "selected_joints" in cfg:
            return list(cfg["selected_joints"])
        raise PoseServiceError(
            code="INVALID_CONFIG",
            message="Config must contain body_joints or selected_joints.",
            status_code=500,
        )

    @staticmethod
    def _body_normalize(coords_33x2: np.ndarray) -> np.ndarray:
        center = (coords_33x2[23] + coords_33x2[24]) / 2.0
        scale = np.linalg.norm(coords_33x2[11] - coords_33x2[12])
        if scale < 1e-6:
            scale = 1e-6
        return (coords_33x2 - center) / scale

    def _preprocess_raw_only(self, raw_frames: np.ndarray, cfg: dict):
        body_joints = self._body_joint_indices(cfg)
        target_frames = int(cfg["target_frames"])

        processed = []
        for coords_33x2 in raw_frames:
            coords = self._body_normalize(coords_33x2)
            coords = coords[body_joints]
            processed.append(coords.flatten())

        frames = np.asarray(processed, dtype=np.float32)
        if len(frames) < 2:
            return None

        x_old = np.linspace(0.0, 1.0, len(frames))
        x_new = np.linspace(0.0, 1.0, target_frames)
        interpolator = interp1d(x_old, frames, axis=0, kind="linear")
        return interpolator(x_new).astype(np.float32)

    def _scale_raw_clip(self, raw_clip: np.ndarray, cfg: dict) -> np.ndarray:
        target_frames = int(cfg["target_frames"])
        n_features = int(cfg["n_features"])

        flat = raw_clip.reshape(1, -1)
        data_min = np.asarray(cfg["scaler_min"], dtype=np.float32)
        data_range = np.asarray(cfg["scaler_data_range"], dtype=np.float32)

        if flat.shape[1] != data_min.shape[0] or flat.shape[1] != data_range.shape[0]:
            raise PoseServiceError(
                code="SCALER_SHAPE_MISMATCH",
                message="Scaler config does not match incoming feature shape.",
                details={
                    "flat_features": int(flat.shape[1]),
                    "scaler_min": int(data_min.shape[0]),
                    "scaler_data_range": int(data_range.shape[0]),
                },
                status_code=500,
            )

        data_range[data_range == 0] = 1.0
        flat_scaled = np.clip((flat - data_min) / data_range, 0.0, 1.0)

        expected = target_frames * n_features
        if flat_scaled.shape[1] != expected:
            raise PoseServiceError(
                code="FEATURE_DIMENSION_MISMATCH",
                message="Config target_frames*n_features does not match processed input.",
                details={
                    "processed_features": int(flat_scaled.shape[1]),
                    "expected_features": int(expected),
                },
                status_code=500,
            )

        return flat_scaled.reshape(1, target_frames, n_features).astype(np.float32)

    def _compute_movement_gate(self, raw_clip: np.ndarray, cfg: dict) -> Tuple[float, float]:
        body_joints = self._body_joint_indices(cfg)

        # Use ArmRaise-style wrist gate when those joints and reference are available.
        if "ref_wrist_y_std" in cfg and 15 in body_joints and 16 in body_joints:
            left_idx = body_joints.index(15) * 2 + 1
            right_idx = body_joints.index(16) * 2 + 1
            wrist_std = float(np.std(raw_clip[:, [left_idx, right_idx]], axis=0).mean())
            ref_std = float(cfg.get("ref_wrist_y_std", 2.9))
            ratio = wrist_std / (ref_std + 1e-6)
            gate = float(np.clip(ratio / 0.3, 0.0, 1.0))
            return gate, ratio

        # Generic fallback gate based on average temporal variation over all features.
        feature_std = float(np.std(raw_clip, axis=0).mean())
        ref_motion_std = float(cfg.get("ref_motion_std", 0.12))
        ratio = feature_std / (ref_motion_std + 1e-6)
        gate = float(np.clip(ratio / 0.3, 0.0, 1.0))
        return gate, ratio

    @staticmethod
    def _tflite_predict(interpreter, input_data: np.ndarray) -> np.ndarray:
        try:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model_input = input_data.astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_details[0]["index"], model_input)
            interpreter.invoke()
            return interpreter.get_tensor(output_details[0]["index"])
        except Exception as exc:
            raise PoseServiceError(
                code="MODEL_INFERENCE_ERROR",
                message="Failed to run TFLite inference for this pose.",
                details={"error": str(exc)},
                status_code=500,
            ) from exc
