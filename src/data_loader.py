"""
src/data_loader.py
各種入力ファイルの読み込みモジュール
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_sync_log(path: str | Path) -> pd.DataFrame:
    """
    sync_log.csv を読み込む。
    カラム: frame_index (int), timestamp_ms (float), led_status (int)
    """
    df = pd.read_csv(path)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    logger.info(f"sync_log 読み込み完了: {len(df)} フレーム")
    return df


def load_sync_params(path: str | Path) -> tuple[float, float]:
    """
    sync_params.json を読み込む。
    変換式: t_rgb [ms] = A * t_event [μs] + B
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    A = float(data["A"])
    B = float(data["B"])
    logger.info(f"sync_params 読み込み完了: A={A}, B={B}")
    return A, B


def load_events(path: str | Path) -> pd.DataFrame:
    """
    events.csv を読み込む。
    先頭のメタデータ行 ('%') をスキップし、
    x, y, polarity, timestamp_us の DataFrame を返す。
    """
    path = Path(path)

    skip_rows = 0
    first_data_line = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("%"):
                skip_rows += 1
            else:
                first_data_line = stripped
                break

    has_header = True
    if first_data_line:
        first_field = first_data_line.split(",")[0].strip()
        try:
            float(first_field)
            has_header = False
        except ValueError:
            has_header = True

    if has_header:
        df = pd.read_csv(path, skiprows=skip_rows)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ("timestamp", "timestamp_us", "t"):
                rename_map[col] = "timestamp_us"
            elif col_lower == "x":
                rename_map[col] = "x"
            elif col_lower == "y":
                rename_map[col] = "y"
            elif col_lower in ("polarity", "p", "pol"):
                rename_map[col] = "polarity"
        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(
            path,
            skiprows=skip_rows,
            header=None,
            names=["x", "y", "polarity", "timestamp_us"],
        )

    df["x"] = df["x"].astype(np.int32)
    df["y"] = df["y"].astype(np.int32)
    df["polarity"] = df["polarity"].astype(np.int32)
    df["timestamp_us"] = df["timestamp_us"].astype(np.float64)

    df = df.sort_values("timestamp_us").reset_index(drop=True)
    logger.info(f"events 読み込み完了: {len(df)} イベント")
    return df


def load_landmarks(path: str | Path) -> dict[int, np.ndarray]:
    """
    landmark.csv を読み込み、フレームごとの 468 頂点座標 (N, 3) の辞書を返す。
    キー: frame_index (int)
    """
    df = pd.read_csv(path)
    df = df[df["face_index"] == 0].copy()

    landmarks_per_frame: dict[int, np.ndarray] = {}
    for frame_idx, group in df.groupby("frame_index"):
        group = group.sort_values("landmark_index")
        group = group[group["landmark_index"] < 468]
        coords = group[["x_norm", "y_norm", "z_norm"]].to_numpy(dtype=np.float64)
        landmarks_per_frame[int(frame_idx)] = coords

    sample_n = next(iter(landmarks_per_frame.values())).shape[0] if landmarks_per_frame else 0
    logger.info(
        f"landmarks 読み込み完了: {len(landmarks_per_frame)} フレーム, "
        f"ランドマーク点数: {sample_n}"
    )
    return landmarks_per_frame


def load_transform_matrix(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    transform_matrix.json を読み込む。
    Returns: (rvec, tvec)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64)
    tvec = np.array(data["tvec"], dtype=np.float64)

    if rvec.ndim == 1:
        rvec = rvec.reshape(3, 1)
    if tvec.ndim == 1:
        tvec = tvec.reshape(3, 1)

    logger.info(f"transform_matrix 読み込み完了: rvec={rvec.ravel()}, tvec={tvec.ravel()}")
    return rvec, tvec


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    calibration.json を読み込む。
    Returns: (intrinsics, distortion)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intrinsics = np.array(data["intrinsics"], dtype=np.float64)
    distortion = np.array(data["distortion"], dtype=np.float64).reshape(1, -1)

    logger.info(f"calibration 読み込み完了: intrinsics shape={intrinsics.shape}")
    return intrinsics, distortion


class RGBFrameReader:
    """
    RGB 動画または画像連番ディレクトリから特定フレームを読み出すヘルパークラス
    """

    def __init__(self, video_path_or_dir: Optional[str] = None):
        self.video_path_or_dir = video_path_or_dir
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_video = False
        self.is_dir = False

        if video_path_or_dir and os.path.exists(video_path_or_dir):
            if os.path.isdir(video_path_or_dir):
                self.is_dir = True
            elif os.path.isfile(video_path_or_dir):
                self.cap = cv2.VideoCapture(video_path_or_dir)
                self.is_video = True

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        frame_index (1-indexed または 0-indexed) の RGB フレーム画像を取得
        """
        if self.is_dir and self.video_path_or_dir:
            # 探索パターン: frame_00001.png, frame_1.png, 1.png, etc.
            candidates = [
                os.path.join(self.video_path_or_dir, f"frame_{frame_index:05d}.png"),
                os.path.join(self.video_path_or_dir, f"frame_{frame_index:04d}.png"),
                os.path.join(self.video_path_or_dir, f"{frame_index:05d}.png"),
                os.path.join(self.video_path_or_dir, f"{frame_index}.png"),
                os.path.join(self.video_path_or_dir, f"frame_{frame_index:05d}_rgb.png"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    return cv2.imread(p)
            return None

        elif self.is_video and self.cap is not None:
            # sync_log.csv は 0-indexed (0 〜 N-1)
            idx = frame_index
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None

        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
