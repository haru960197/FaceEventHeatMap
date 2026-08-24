"""
src/uv_buffer.py
UVポジションバッファ生成モジュール

各RGBフレームの顔ランドマーク（468点）とカメラパラメータから、
イベントカメラ画像平面 (320x320) の各ピクセルに対応する
1. 標準顔テンプレート (Canonical 2D / UV) 座標 (u_template, v_template) in [0, 1]
2. RGBカメラ画像平面でのピクセル座標 (u_rgb, v_rgb)
を格納したバッファを事前生成する。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _transform_landmarks_to_camera_coords(
    landmarks: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    ランドマーク点群 (N, 3) をイベントカメラ座標系 3D 点に変換する。
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    cam_coords = (R @ landmarks.T).T + t
    return cam_coords


def _project_to_pixel(
    cam_coords_3d: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """
    イベントカメラ座標系 3D 点をイベントカメラのピクセル座標 (u, v) に投影する。
    """
    pts = cam_coords_3d.reshape(-1, 1, 3).astype(np.float64)
    rvec_zero = np.zeros((3, 1), dtype=np.float64)
    tvec_zero = np.zeros((3, 1), dtype=np.float64)

    projected, _ = cv2.projectPoints(
        pts,
        rvec_zero,
        tvec_zero,
        intrinsics,
        distortion,
    )
    return projected.reshape(-1, 2)


def build_frame_buffers(
    landmarks: np.ndarray,
    template_coords_norm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    triangles: np.ndarray,
    image_width: int = 320,
    image_height: int = 320,
    rgb_width: int = 1920,
    rgb_height: int = 1080,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    単一フレームのランドマークから、
    1. template_uv_buffer [image_height, image_width, 2]
    2. rgb_pixel_buffer [image_height, image_width, 2]
    を生成する。

    Parameters
    ----------
    landmarks : np.ndarray
        shape (468, 3) のRGBカメラ正規化ランドマーク (x_norm, y_norm, z_norm)
    template_coords_norm : np.ndarray
        shape (468, 2) の標準顔上の2D正規化座標 [0, 1]
    rvec, tvec : np.ndarray
        PnP外部パラメータ
    intrinsics, distortion : np.ndarray
        イベントカメラ内部パラメータ
    triangles : np.ndarray
        shape (N_triangles, 3) の三角形インデックス
    image_width, image_height : int
        イベントカメラ解像度 (320, 320)
    rgb_width, rgb_height : int
        RGBカメラ解像度 (1920, 1080)

    Returns
    -------
    template_uv_buffer : np.ndarray (H, W, 2)
        各ピクセルの標準顔テンプレート正規化座標 (u, v)。顔外は NaN。
    rgb_pixel_buffer : np.ndarray (H, W, 2)
        各ピクセルのRGB画像ピクセル座標 (u_rgb, v_rgb)。顔外は NaN。
    """
    # 1. ランドマーク -> イベントカメラ座標系 3D 点
    cam_xyz = _transform_landmarks_to_camera_coords(landmarks, rvec, tvec)

    # 2. 3D 点 -> イベントカメラピクセル座標 (u, v)
    pixels_uv = _project_to_pixel(cam_xyz, intrinsics, distortion)

    # RGB 画像上のランドマークピクセル座標
    rgb_landmarks_px = landmarks[:, :2] * np.array([rgb_width, rgb_height], dtype=np.float32)

    # バッファ初期化
    z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)
    template_uv_buffer = np.full((image_height, image_width, 2), np.nan, dtype=np.float32)
    rgb_pixel_buffer = np.full((image_height, image_width, 2), np.nan, dtype=np.float32)

    # ソフトウェアラスタライザ
    grid_x, grid_y = np.meshgrid(
        np.arange(image_width, dtype=np.float32),
        np.arange(image_height, dtype=np.float32),
    )

    for tri in triangles:
        i0, i1, i2 = tri
        A3, B3, C3 = cam_xyz[i0], cam_xyz[i1], cam_xyz[i2]

        u0, v0 = pixels_uv[i0]
        u1, v1 = pixels_uv[i1]
        u2, v2 = pixels_uv[i2]

        # 2D Bounding Box
        min_x = max(0, int(np.floor(min(u0, u1, u2))))
        max_x = min(image_width - 1, int(np.ceil(max(u0, u1, u2))))
        min_y = max(0, int(np.floor(min(v0, v1, v2))))
        max_y = min(image_height - 1, int(np.ceil(max(v0, v1, v2))))

        if min_x > max_x or min_y > max_y:
            continue

        xx = grid_x[min_y : max_y + 1, min_x : max_x + 1]
        yy = grid_y[min_y : max_y + 1, min_x : max_x + 1]

        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < 1e-6:
            continue

        w0 = ((v1 - v2) * (xx - u2) + (u2 - u1) * (yy - v2)) / denom
        w1 = ((v2 - v0) * (xx - u2) + (u0 - u2) * (yy - v2)) / denom
        w2 = 1.0 - w0 - w1

        inside_mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside_mask):
            continue

        w0_in = w0[inside_mask]
        w1_in = w1[inside_mask]
        w2_in = w2[inside_mask]

        # 補間深度 (Z の絶対値)
        z_interp = w0_in * A3[2] + w1_in * B3[2] + w2_in * C3[2]
        depth = np.abs(z_interp)

        current_z_bbox = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        current_z_in = current_z_bbox[inside_mask]

        # Zバッファ比較: 手前のポリゴンを優先
        update_mask = depth < current_z_in
        if not np.any(update_mask):
            continue

        # 更新が必要なピクセルの重み
        w0_up = w0_in[update_mask]
        w1_up = w1_in[update_mask]
        w2_up = w2_in[update_mask]

        # 標準顔テンプレート座標の補間
        uv0, uv1, uv2 = (
            template_coords_norm[i0],
            template_coords_norm[i1],
            template_coords_norm[i2],
        )
        u_tmpl = w0_up * uv0[0] + w1_up * uv1[0] + w2_up * uv2[0]
        v_tmpl = w0_up * uv0[1] + w1_up * uv1[1] + w2_up * uv2[1]

        # RGB 画像ピクセル座標の補間
        rgb0, rgb1, rgb2 = (
            rgb_landmarks_px[i0],
            rgb_landmarks_px[i1],
            rgb_landmarks_px[i2],
        )
        u_rgb = w0_up * rgb0[0] + w1_up * rgb1[0] + w2_up * rgb2[0]
        v_rgb = w0_up * rgb0[1] + w1_up * rgb1[1] + w2_up * rgb2[1]

        # バッファへの書き込み
        # inside_mask の中で update_mask が True のインデックス
        inside_indices = np.where(inside_mask)
        update_y_indices = inside_indices[0][update_mask] + min_y
        update_x_indices = inside_indices[1][update_mask] + min_x

        z_buffer[update_y_indices, update_x_indices] = depth[update_mask]
        template_uv_buffer[update_y_indices, update_x_indices, 0] = u_tmpl
        template_uv_buffer[update_y_indices, update_x_indices, 1] = v_tmpl
        rgb_pixel_buffer[update_y_indices, update_x_indices, 0] = u_rgb
        rgb_pixel_buffer[update_y_indices, update_x_indices, 1] = v_rgb

    return template_uv_buffer, rgb_pixel_buffer
