"""
src/heatmap_accumulator.py
イベント累積・ヒートマップ集計モジュール

時間窓 [t - M, t] (ms) 内のイベントを抽出し、
1. 標準顔テンプレート (UV) 空間
2. RGB 画像空間
3. イベントカメラ視点空間
の 2D 配列にヒストグラム累積を行う。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HeatmapAccumulator:
    """
    イベントの空間マッピングおよび 2D カウント累積を管理するクラス
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        sync_A: float,
        sync_B: float,
        uv_width: int = 512,
        uv_height: int = 512,
        rgb_width: int = 1920,
        rgb_height: int = 1080,
    ):
        """
        Parameters
        ----------
        events_df : pd.DataFrame
            x, y, polarity, timestamp_us を含む DataFrame
        sync_A, sync_B : float
            t_rgb [ms] = sync_A * t_event [us] + sync_B
        """
        self.events_df = events_df
        self.sync_A = sync_A
        self.sync_B = sync_B
        self.uv_width = uv_width
        self.uv_height = uv_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        # イベントのタイムスタンプを RGB 時間 (ms) に変換して保持
        self.x_coords = events_df["x"].to_numpy(dtype=np.int32)
        self.y_coords = events_df["y"].to_numpy(dtype=np.int32)
        self.polarities = events_df["polarity"].to_numpy(dtype=np.int32)
        self.timestamps_us = events_df["timestamp_us"].to_numpy(dtype=np.float64)
        
        # t_rgb = A * t_event + B
        self.timestamps_rgb_ms = self.sync_A * self.timestamps_us + self.sync_B

        # 総累積（全シーケンス）用のカウント配列
        self.total_uv_counts = {
            "all": np.zeros((uv_height, uv_width), dtype=np.float32),
            "pol0": np.zeros((uv_height, uv_width), dtype=np.float32),
            "pol1": np.zeros((uv_height, uv_width), dtype=np.float32),
        }

    def get_events_in_window(
        self, t_end_ms: float, window_ms: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        [t_end_ms - window_ms, t_end_ms] に収まるイベント群のインデックスを抽出する。

        Returns
        -------
        x, y, pol, t_ms : np.ndarray
        """
        t_start_ms = t_end_ms - window_ms
        idx_start = np.searchsorted(self.timestamps_rgb_ms, t_start_ms, side="left")
        idx_end = np.searchsorted(self.timestamps_rgb_ms, t_end_ms, side="right")

        if idx_start >= idx_end:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )

        return (
            self.x_coords[idx_start:idx_end],
            self.y_coords[idx_start:idx_end],
            self.polarities[idx_start:idx_end],
            self.timestamps_rgb_ms[idx_start:idx_end],
        )

    def accumulate_uv_heatmap(
        self,
        ev_x: np.ndarray,
        ev_y: np.ndarray,
        ev_pol: np.ndarray,
        uv_buffer: np.ndarray,
        update_total: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        指定したイベント群と uv_buffer から、標準顔 UV 空間のカウントマップを生成する。

        Parameters
        ----------
        ev_x, ev_y, ev_pol : np.ndarray
            イベント座標および極性
        uv_buffer : np.ndarray (H_evt, W_evt, 2)
            イベントカメラ画像平面各ピクセルの正規化 UV 座標 [0, 1]。顔外は NaN。
        update_total : bool
            総累積マップにも加算するかどうか

        Returns
        -------
        dict: {"all": array(H, W), "pol0": array(H, W), "pol1": array(H, W)}
        """
        counts = {
            "all": np.zeros((self.uv_height, self.uv_width), dtype=np.float32),
            "pol0": np.zeros((self.uv_height, self.uv_width), dtype=np.float32),
            "pol1": np.zeros((self.uv_height, self.uv_width), dtype=np.float32),
        }

        if len(ev_x) == 0:
            return counts

        # 各イベントの (x, y) に対応する UV 座標を取得
        # 有効範囲内チェック
        h_evt, w_evt = uv_buffer.shape[:2]
        valid_coords = (ev_x >= 0) & (ev_x < w_evt) & (ev_y >= 0) & (ev_y < h_evt)
        x_v = ev_x[valid_coords]
        y_v = ev_y[valid_coords]
        pol_v = ev_pol[valid_coords]

        # UV 座標の参照
        uvs = uv_buffer[y_v, x_v]  # shape (N, 2)

        # NaN（顔領域外）を除外
        valid_face = ~np.isnan(uvs[:, 0]) & ~np.isnan(uvs[:, 1])
        if not np.any(valid_face):
            return counts

        u_face = uvs[valid_face, 0]
        v_face = uvs[valid_face, 1]
        pol_face = pol_v[valid_face]

        # UV ピクセル座標に変換 [0, width-1], [0, height-1]
        px_u = np.clip((u_face * (self.uv_width - 1)).astype(np.int32), 0, self.uv_width - 1)
        px_v = np.clip((v_face * (self.uv_height - 1)).astype(np.int32), 0, self.uv_height - 1)

        # 2D ヒストグラムの高速累積 (np.add.at)
        np.add.at(counts["all"], (px_v, px_u), 1.0)

        mask_pol0 = pol_face == 0
        if np.any(mask_pol0):
            np.add.at(counts["pol0"], (px_v[mask_pol0], px_u[mask_pol0]), 1.0)

        mask_pol1 = pol_face == 1
        if np.any(mask_pol1):
            np.add.at(counts["pol1"], (px_v[mask_pol1], px_u[mask_pol1]), 1.0)

        if update_total:
            self.total_uv_counts["all"] += counts["all"]
            self.total_uv_counts["pol0"] += counts["pol0"]
            self.total_uv_counts["pol1"] += counts["pol1"]

        return counts

    def accumulate_rgb_heatmap(
        self,
        ev_x: np.ndarray,
        ev_y: np.ndarray,
        ev_pol: np.ndarray,
        rgb_pixel_buffer: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        指定したイベント群と rgb_pixel_buffer から、RGB 画像平面上のカウントマップを生成する。
        """
        counts = {
            "all": np.zeros((self.rgb_height, self.rgb_width), dtype=np.float32),
            "pol0": np.zeros((self.rgb_height, self.rgb_width), dtype=np.float32),
            "pol1": np.zeros((self.rgb_height, self.rgb_width), dtype=np.float32),
        }

        if len(ev_x) == 0:
            return counts

        h_evt, w_evt = rgb_pixel_buffer.shape[:2]
        valid_coords = (ev_x >= 0) & (ev_x < w_evt) & (ev_y >= 0) & (ev_y < h_evt)
        x_v = ev_x[valid_coords]
        y_v = ev_y[valid_coords]
        pol_v = ev_pol[valid_coords]

        rgb_pts = rgb_pixel_buffer[y_v, x_v]  # shape (N, 2)
        valid_face = ~np.isnan(rgb_pts[:, 0]) & ~np.isnan(rgb_pts[:, 1])

        if not np.any(valid_face):
            return counts

        u_rgb = rgb_pts[valid_face, 0]
        v_rgb = rgb_pts[valid_face, 1]
        pol_face = pol_v[valid_face]

        px_u = np.clip(np.round(u_rgb).astype(np.int32), 0, self.rgb_width - 1)
        px_v = np.clip(np.round(v_rgb).astype(np.int32), 0, self.rgb_height - 1)

        np.add.at(counts["all"], (px_v, px_u), 1.0)

        mask_pol0 = pol_face == 0
        if np.any(mask_pol0):
            np.add.at(counts["pol0"], (px_v[mask_pol0], px_u[mask_pol0]), 1.0)

        mask_pol1 = pol_face == 1
        if np.any(mask_pol1):
            np.add.at(counts["pol1"], (px_v[mask_pol1], px_u[mask_pol1]), 1.0)

        return counts
