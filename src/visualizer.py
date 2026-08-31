"""
src/visualizer.py
ヒートマップ可視化・レンダリング・動画出力モジュール

2D カウントマップに対して、
1. ガウシアンブラー平滑化
2. パーセンタイル / 最大値正規化
3. カラーマップ適用 (turbo, viridis, jet, inferno, etc.)
4. 背景（標準顔テンプレート / RGBフレーム）とのアルファブレンド合成
5. カラーバー / 情報テキスト描画
6. 画像 / 動画保存
を行う。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger(__name__)


def apply_gaussian_blur(count_map: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    カウントマップにガウシアンブラーを適用して滑らかな密度マップにする。
    sigma <= 0 の場合はそのまま返す。
    """
    if sigma <= 0:
        return count_map.copy()

    # kernel size を sigma から自動決定 (奇数)
    ksize = int(np.ceil(sigma * 3.0)) * 2 + 1
    blurred = cv2.GaussianBlur(count_map, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred


def normalize_heatmap(
    density_map: np.ndarray,
    mode: str = "percentile",
    percentile_val: float = 98.5,
    min_threshold: float = 0.05,
    fixed_vmax: Optional[float] = None,
    scale_type: str = "sqrt",
    min_vmax_floor: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    密度マップを [0.0, 1.0] の範囲に正規化する。

    Parameters
    ----------
    density_map : np.ndarray
        平滑化済みの密度配列 (H, W)
    mode : str
        "percentile", "max", "fixed"
    percentile_val : float
        パーセンタイル値 (例: 98.5)
    min_threshold : float
        最小表示閾値（これ以下の値は 0.0 にカット）
    fixed_vmax : float, optional
        mode="fixed" の場合の最大値
    scale_type : str
        "linear" (線形), "sqrt" (平方根圧縮), "log" (対数圧縮)
    min_vmax_floor : float
        静止フレームで vmax が不当に小さくなるのを防ぐ下限基準値 (events/px)

    Returns
    -------
    norm_map : np.ndarray in [0.0, 1.0]
    vmax : float (正規化に使用された最大基準値)
    """
    non_zero = density_map[density_map > 0]
    if len(non_zero) == 0:
        return np.zeros_like(density_map), 1.0

    if mode == "fixed" and fixed_vmax is not None and fixed_vmax > 0:
        vmax = float(fixed_vmax)
    elif mode == "percentile":
        raw_vmax = float(np.percentile(non_zero, percentile_val))
        vmax = max(raw_vmax, float(min_vmax_floor))
        if vmax <= 0:
            vmax = float(np.max(non_zero))
    else:  # max
        raw_vmax = float(np.max(non_zero))
        vmax = max(raw_vmax, float(min_vmax_floor))

    if vmax <= 0:
        vmax = 1.0

    # スケール変換 (linear, sqrt, log)
    clamped_density = np.maximum(density_map, 0.0)
    if scale_type == "log":
        # log(1 + d) / log(1 + vmax)
        norm_map = np.log1p(clamped_density) / np.log1p(vmax)
    elif scale_type == "sqrt":
        # sqrt(d) / sqrt(vmax)
        norm_map = np.sqrt(clamped_density) / np.sqrt(vmax)
    else:  # linear
        norm_map = clamped_density / vmax

    norm_map = np.clip(norm_map, 0.0, 1.0)

    # 閾値処理: min_threshold 未満は 0.0 にして背景を見やすく
    norm_map[norm_map < min_threshold] = 0.0

    return norm_map, vmax


def colorize_heatmap(
    norm_map: np.ndarray,
    colormap_name: str = "turbo",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    正規化された [0.0, 1.0] 配列にカラーマップを適用し、
    BGR 画像 (H, W, 3) および アルファマスク (H, W) を返す。
    """
    # Matplotlib のカラーマップを取得
    try:
        cmap = cm.get_cmap(colormap_name)
    except (ValueError, AttributeError):
        cmap = cm.turbo

    # RGBA (H, W, 4) in [0.0, 1.0]
    rgba = cmap(norm_map)

    # BGR (H, W, 3) in [0, 255]
    bgr = (rgba[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)

    # アルファマスク (H, W): norm_map が 0 の場所は 0.0, それ以外は値に応じた不透明度
    alpha_mask = norm_map.astype(np.float32)

    return bgr, alpha_mask


def blend_heatmap_with_background(
    background_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    global_alpha: float = 0.7,
) -> np.ndarray:
    """
    背景画像とカラー化されたヒートマップを半透明合成する。

    Parameters
    ----------
    background_bgr : np.ndarray (H, W, 3) uint8
    heatmap_bgr : np.ndarray (H, W, 3) uint8
    alpha_mask : np.ndarray (H, W) float32 in [0, 1]
    global_alpha : float
        ヒートマップ全体のブレンド強度

    Returns
    -------
    blended_bgr : np.ndarray (H, W, 3) uint8
    """
    bg = background_bgr.astype(np.float32)
    hm = heatmap_bgr.astype(np.float32)

    # 各ピクセルの合成アルファ
    a = (alpha_mask * global_alpha)[:, :, np.newaxis]

    out = hm * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_colorbar(
    image_bgr: np.ndarray,
    vmax: float,
    colormap_name: str = "turbo",
    label: str = "Events / pixel",
    bar_width: int = 16,
    bar_height: int = 150,
    margin_right: int = 25,
    margin_top: int = 40,
) -> np.ndarray:
    """
    画像の右上にカラーバー（凡例）を描画する。
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    x1 = w - margin_right - bar_width
    x2 = w - margin_right
    y1 = margin_top
    y2 = margin_top + bar_height

    if x1 < 0 or y2 > h:
        return img

    # カラーグラデーション作成 (上: 1.0, 下: 0.0)
    gradient = np.linspace(1.0, 0.0, bar_height, dtype=np.float32)
    grad_map = np.tile(gradient[:, np.newaxis], (1, bar_width))

    try:
        cmap = cm.get_cmap(colormap_name)
    except (ValueError, AttributeError):
        cmap = cm.turbo

    grad_rgba = cmap(grad_map)
    grad_bgr = (grad_rgba[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)

    # カラーバー貼り付け
    img[y1:y2, x1:x2] = grad_bgr
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2, y2), (200, 200, 200), 1)

    # ラベル・目盛り文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    color = (240, 240, 240)
    thick = 1

    # 最大値
    max_str = f"{vmax:.1f}" if vmax < 10 else f"{int(vmax)}"
    cv2.putText(img, max_str, (x1 - 38, y1 + 10), font, font_scale, color, thick, cv2.LINE_AA)

    # 最小値 (0)
    cv2.putText(img, "0", (x1 - 18, y2), font, font_scale, color, thick, cv2.LINE_AA)

    # タイトルラベル
    cv2.putText(img, label, (x1 - 60, y1 - 12), font, font_scale, color, thick, cv2.LINE_AA)

    return img


def draw_info_overlay(
    image_bgr: np.ndarray,
    frame_idx: int,
    t_ms: float,
    window_ms: float,
    event_count: int,
    extra_text: Optional[str] = None,
) -> np.ndarray:
    """
    画像の左上/左下に情報テキストを描画する。
    """
    img = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (255, 255, 255)
    bg_box_color = (0, 0, 0)
    thick = 1

    lines = [
        f"Frame: {frame_idx:05d}",
        f"Time: {t_ms:0.1f} ms",
        f"Window: [{t_ms - window_ms:0.1f}, {t_ms:0.1f}] ms ({window_ms:0.0f}ms)",
        f"Events: {event_count:,}",
    ]
    if extra_text:
        lines.append(extra_text)

    # 半透明ボックス
    box_h = len(lines) * 20 + 12
    box_w = 260
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), bg_box_color, -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    y = 26
    for line in lines:
        cv2.putText(img, line, (16, y), font, font_scale, color, thick, cv2.LINE_AA)
        y += 20

    return img
