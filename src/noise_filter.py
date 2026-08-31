"""
src/noise_filter.py
イベントカメラのホットピクセル（異常振動・異常発火）およびノイズ抑制モジュール

1. 統計的ホットピクセル検出 (Static / Statistical Hot Pixel Detection)
   - 局所近傍比率 (Spatial Outlier Ratio)
   - 許容最大発火率 (Max Event Rate in Hz)
   - センサー境界マージン (Border Margin)
2. 不応期フィルタ (Refractory Period Filter)
   - 同一ピクセルにおける極短時間連続イベントの抑制
3. 空間メディアンフィルタ (Spatial Median Filter)
   - 2D カウントマップ平滑化前の孤立スパイク除去
4. ホットピクセル可視化 & レポート出力
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def detect_hot_pixels(
    events_df: pd.DataFrame,
    image_width: int = 320,
    image_height: int = 320,
    ratio_threshold: float = 3.5,
    min_count: int = 500,
    max_rate_hz: float = 800.0,
    border_margin: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    全イベントの空間統計からホットピクセル（異常発火ピクセル）を自動検出する。

    Parameters
    ----------
    events_df : pd.DataFrame
        x, y, polarity, timestamp_us を含むイベント DataFrame
    image_width : int
        イベントカメラの横解像度 (px)
    image_height : int
        イベントカメラの縦解像度 (px)
    ratio_threshold : float
        近傍平均発火数に対する比率閾値 (この倍率以上でホット判定)
    min_count : int
        判定対象とする最小累積イベント数
    max_rate_hz : float
        許容最大発火率 (Hz)
    border_margin : int
        センサー最外周のマージン幅 (px)。この範囲内のピクセルは無条件でホット扱いとして除外。

    Returns
    -------
    hot_mask : np.ndarray (H, W) bool
        ホットピクセルの Boolean マスク (True: ホットピクセル)
    stats : dict
        検出統計情報 (総ピクセル数, ホットピクセル数, 理由別内訳など)
    """
    event_counts = np.zeros((image_height, image_width), dtype=np.float64)
    x_coords = events_df["x"].to_numpy(dtype=np.int32)
    y_coords = events_df["y"].to_numpy(dtype=np.int32)

    # 座標の有効範囲チェック
    valid = (x_coords >= 0) & (x_coords < image_width) & (y_coords >= 0) & (y_coords < image_height)
    x_valid = x_coords[valid]
    y_valid = y_coords[valid]

    np.add.at(event_counts, (y_valid, x_valid), 1.0)

    # 全体期間 (秒)
    if len(events_df) > 1:
        t_min = float(events_df["timestamp_us"].min())
        t_max = float(events_df["timestamp_us"].max())
        duration_sec = max((t_max - t_min) / 1e6, 0.001)
    else:
        duration_sec = 1.0

    rate_hz = event_counts / duration_sec

    # 3x3 近傍平均（中心ピクセルを除外）
    mean_3x3 = uniform_filter(event_counts, size=3)
    neighbor_mean = (9.0 * mean_3x3 - event_counts) / 8.0

    # 近傍比率の計算
    ratio = np.zeros_like(event_counts)
    active_mask = event_counts >= min_count
    ratio[active_mask] = event_counts[active_mask] / np.maximum(neighbor_mean[active_mask], 1.0)

    # 判定条件: 3x3 近傍比率 または 最大レート
    ratio_hot = (ratio >= ratio_threshold) & (event_counts >= min_count)
    rate_hot = rate_hz >= max_rate_hz

    # 境界マージン
    border_hot = np.zeros((image_height, image_width), dtype=bool)
    if border_margin > 0:
        border_hot[:border_margin, :] = True
        border_hot[-border_margin:, :] = True
        border_hot[:, :border_margin] = True
        border_hot[:, -border_margin:] = True

    hot_mask = ratio_hot | rate_hot | border_hot

    num_total_pixels = image_width * image_height
    num_hot_pixels = int(np.sum(hot_mask))

    stats = {
        "total_pixels": num_total_pixels,
        "hot_pixels": num_hot_pixels,
        "hot_pixel_ratio_pct": (num_hot_pixels / num_total_pixels) * 100.0,
        "reason_ratio_count": int(np.sum(ratio_hot)),
        "reason_rate_count": int(np.sum(rate_hot)),
        "reason_border_count": int(np.sum(border_hot)),
        "duration_sec": duration_sec,
        "max_pixel_events": int(np.max(event_counts)),
        "event_counts_2d": event_counts,
        "ratio_2d": ratio,
        "rate_hz_2d": rate_hz,
    }

    logger.info(
        f"[HotPixelFilter] Detected {num_hot_pixels}/{num_total_pixels} hot pixels "
        f"({stats['hot_pixel_ratio_pct']:.2f}%). "
        f"[Ratio >= {ratio_threshold}: {stats['reason_ratio_count']}, "
        f"Rate >= {max_rate_hz}Hz: {stats['reason_rate_count']}, "
        f"Border (margin={border_margin}): {stats['reason_border_count']}]"
    )

    return hot_mask, stats


def filter_events_by_hot_pixels(
    events_df: pd.DataFrame,
    hot_mask: np.ndarray,
) -> Tuple[pd.DataFrame, int, float]:
    """
    ホットピクセルマスクに基づいて、ホットピクセルから発生したイベントを除外する。

    Parameters
    ----------
    events_df : pd.DataFrame
    hot_mask : np.ndarray (H, W) bool

    Returns
    -------
    filtered_df : pd.DataFrame
    num_removed : int
    removed_pct : float
    """
    x_coords = events_df["x"].to_numpy(dtype=np.int32)
    y_coords = events_df["y"].to_numpy(dtype=np.int32)
    h, w = hot_mask.shape

    valid_coords = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    is_hot = np.zeros(len(events_df), dtype=bool)
    is_hot[valid_coords] = hot_mask[y_coords[valid_coords], x_coords[valid_coords]]

    num_before = len(events_df)
    filtered_df = events_df[~is_hot].reset_index(drop=True)
    num_after = len(filtered_df)
    num_removed = num_before - num_after
    removed_pct = (num_removed / num_before * 100.0) if num_before > 0 else 0.0

    logger.info(
        f"[HotPixelFilter] Filtered events: {num_after:,} / {num_before:,} remaining "
        f"({num_removed:,} removed, {removed_pct:.2f}%)"
    )

    return filtered_df, num_removed, removed_pct


def apply_refractory_filter(
    events_df: pd.DataFrame,
    refractory_period_us: float = 500.0,
    image_width: int = 320,
    image_height: int = 320,
) -> Tuple[pd.DataFrame, int, float]:
    """
    不応期フィルタ（Refractory Period Filter）。
    同一ピクセルにおいて直前のイベントとの時間差が refractory_period_us 未満のイベントを除去する。

    Parameters
    ----------
    events_df : pd.DataFrame
        timestamp_us 昇順にソートされていること
    refractory_period_us : float
        不応期 (マイクロ秒)
    image_width, image_height : int

    Returns
    -------
    filtered_df : pd.DataFrame
    num_removed : int
    removed_pct : float
    """
    if refractory_period_us <= 0 or len(events_df) == 0:
        return events_df, 0, 0.0

    x_arr = events_df["x"].to_numpy(dtype=np.int32)
    y_arr = events_df["y"].to_numpy(dtype=np.int32)
    t_arr = events_df["timestamp_us"].to_numpy(dtype=np.float64)

    # 1D ピクセルインデックス (0 〜 H*W-1)
    pixel_idx = y_arr * image_width + x_arr
    total_pixels = image_width * image_height

    # ピクセルごとの直前発火時刻を保持する配列 (初期値: -inf)
    last_timestamp = np.full(total_pixels, -1e12, dtype=np.float64)

    keep_mask = np.ones(len(events_df), dtype=bool)

    # 高速反復判定
    for i in range(len(events_df)):
        pid = pixel_idx[i]
        if 0 <= pid < total_pixels:
            t = t_arr[i]
            if t - last_timestamp[pid] < refractory_period_us:
                keep_mask[i] = False
            else:
                last_timestamp[pid] = t

    num_before = len(events_df)
    filtered_df = events_df[keep_mask].reset_index(drop=True)
    num_after = len(filtered_df)
    num_removed = num_before - num_after
    removed_pct = (num_removed / num_before * 100.0) if num_before > 0 else 0.0

    logger.info(
        f"[RefractoryFilter] Filtered events (period={refractory_period_us:.1f}us): "
        f"{num_after:,} / {num_before:,} remaining ({num_removed:,} removed, {removed_pct:.2f}%)"
    )

    return filtered_df, num_removed, removed_pct


def apply_spatial_median_filter(
    count_map: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    2D カウントマップにメディアンフィルタを適用し、孤立スパイクを除去する。
    ksize <= 1 の場合はそのまま返す。

    Parameters
    ----------
    count_map : np.ndarray (H, W) float32
    ksize : int (奇数: 3, 5 等)

    Returns
    -------
    filtered_map : np.ndarray
    """
    if ksize <= 1:
        return count_map.copy()

    if ksize % 2 == 0:
        ksize += 1

    # OpenCV medianBlur は float32 配列に対応
    filtered = cv2.medianBlur(count_map.astype(np.float32), ksize)
    return filtered


def save_hot_pixel_visualization(
    hot_mask: np.ndarray,
    event_counts_2d: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    検出されたホットピクセルの位置と発火数を可視化した画像を保存する。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = hot_mask.shape
    # 背景: 発火数の対数スケールグレースケール
    log_counts = np.log1p(event_counts_2d)
    max_val = np.max(log_counts) if np.max(log_counts) > 0 else 1.0
    gray = (log_counts / max_val * 200).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ホットピクセルを赤色 (0, 0, 255) で強調
    bgr[hot_mask] = [0, 0, 255]

    # 解像度が小さい場合は拡大表示
    if w < 512 or h < 512:
        scale = max(512 // w, 512 // h, 1)
        bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(output_path), bgr)
    logger.info(f"[HotPixelFilter] Saved hot pixel mask visualization to {output_path}")


def save_hot_pixel_report(
    hot_mask: np.ndarray,
    stats: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    ホットピクセル一覧の詳細（座標, 発火数, 近傍比率, 発火率Hz）を CSV ファイルに保存する。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_counts = stats.get("event_counts_2d")
    ratio = stats.get("ratio_2d")
    rate_hz = stats.get("rate_hz_2d")

    coords = np.argwhere(hot_mask)
    rows = []
    for y, x in coords:
        c = int(event_counts[y, x]) if event_counts is not None else 0
        r = float(ratio[y, x]) if ratio is not None else 0.0
        hz = float(rate_hz[y, x]) if rate_hz is not None else 0.0
        rows.append({"x": int(x), "y": int(y), "event_count": c, "neighbor_ratio": r, "rate_hz": hz})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("event_count", ascending=False).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"[HotPixelFilter] Saved hot pixel report ({len(df)} pixels) to {output_path}")
