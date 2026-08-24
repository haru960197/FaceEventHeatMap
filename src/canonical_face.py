"""
src/canonical_face.py
MediaPipe Canonical Face Model (標準顔モデル) の定義・座標管理モジュール

MediaPipe の Canonical Face Model (468頂点) から:
- 3D Metric 座標 (X, Y, Z)
- 2D 正面正規化座標 (x_2d, y_2d) in [0, 1]
- UV テクスチャ展開座標 (u, v) in [0, 1]
- 三角形テッセレーション (Face Mesh Triangles)
を取得・生成・キャッシュし、背景用テンプレート画像を生成する。
"""

from __future__ import annotations

import logging
import os
import urllib.request
from collections import defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Canonical Face Model OBJ URL
CANONICAL_OBJ_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)


def get_face_triangles() -> np.ndarray:
    """
    MediaPipe 0.10.x の Tasks API から FACE_LANDMARKS_TESSELATION を取得し、
    三角形インデックスリスト (N_triangles, 3) を構築する。

    Returns
    -------
    np.ndarray
        shape (N_triangles, 3) の整数配列。各行が1つの三角形の3頂点インデックス (0〜467)。
    """
    try:
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarksConnections,
        )
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    except (ImportError, AttributeError):
        # 万が一 Tasks API が使えない場合のフォールバック（旧API）
        import mediapipe as mp
        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

    edges = [(c.start, c.end) for c in connections]

    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    triangles = []
    visited: set[tuple[int, int, int]] = set()
    for u, v in edges:
        common = adj[u] & adj[v]
        for w in common:
            tri = tuple(sorted([u, v, w]))
            if tri not in visited:
                visited.add(tri)
                triangles.append(tri)

    tri_array = np.array(triangles, dtype=np.int32)
    return tri_array


def download_canonical_obj(save_path: str) -> None:
    """Canonical Face Model OBJ ファイルをダウンロードする"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        logger.info(f"Downloading canonical face model OBJ to {save_path}...")
        urllib.request.urlretrieve(CANONICAL_OBJ_URL, save_path)
        logger.info("Download completed.")


def parse_canonical_obj(obj_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    canonical_face_model.obj をパースして、
    1. 468頂点の3D座標 (468, 3)
    2. 468頂点のUV座標 (468, 2)
    3. 468頂点の正面2D正規化座標 (468, 2)
    を抽出する。

    Returns
    -------
    coords_3d : np.ndarray (468, 3)
    uv_coords : np.ndarray (468, 2)
    coords_2d_front : np.ndarray (468, 2)
    """
    vertices = []
    uvs_raw = []
    # 各頂点 v_idx (1-based) に対応する vt_idx (1-based) を記録
    v_to_vt = {}

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                # 3D 頂点
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt":
                # テクスチャ UV
                uvs_raw.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                # 面: v/vt/vn
                for face_part in parts[1:]:
                    vals = face_part.split("/")
                    v_idx = int(vals[0]) - 1
                    if len(vals) > 1 and vals[1]:
                        vt_idx = int(vals[1]) - 1
                        if v_idx not in v_to_vt and vt_idx < len(uvs_raw):
                            v_to_vt[v_idx] = vt_idx

    coords_3d = np.array(vertices[:468], dtype=np.float32)

    # UV 座標の割り当て (468, 2)
    uv_coords = np.zeros((468, 2), dtype=np.float32)
    uv_raw_arr = np.array(uvs_raw, dtype=np.float32) if uvs_raw else None

    for i in range(468):
        if i in v_to_vt and uv_raw_arr is not None:
            vt_idx = v_to_vt[i]
            # OBJ の V 座標は通常下原点なので、画像系に合わせて反転 (1.0 - v)
            u = uv_raw_arr[vt_idx, 0]
            v = 1.0 - uv_raw_arr[vt_idx, 1]
            uv_coords[i] = [u, v]
        else:
            # フォールバック: 3D 座標の XY を用いる
            uv_coords[i] = [0.5, 0.5]

    # 正面 2D 座標 (468, 2): X はそのまま, Y は反転（上が 0, 下が 1 の画像座標系に合わせる）
    # 顔中心を (0.5, 0.5) に配置し、適度なマージン (例: 80%幅) を持たせて正規化
    x_raw = coords_3d[:, 0]
    y_raw = -coords_3d[:, 1]  # Y 反転 (顔上部を上方向へ)

    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    span = max(x_max - x_min, y_max - y_min) * 1.15  # 15% マージン

    norm_x = (x_raw - cx) / span + 0.5
    norm_y = (y_raw - cy) / span + 0.5
    coords_2d_front = np.stack([norm_x, norm_y], axis=1).astype(np.float32)

    return coords_3d, uv_coords, coords_2d_front


class CanonicalFaceModel:
    """
    Canonical Face Model のデータ管理クラス
    """

    def __init__(self, asset_dir: str = "assets"):
        self.asset_dir = asset_dir
        os.makedirs(asset_dir, exist_ok=True)
        self.obj_path = os.path.join(asset_dir, "canonical_face_model.obj")
        
        # OBJ が無ければダウンロード
        download_canonical_obj(self.obj_path)

        # パース
        self.coords_3d, self.uv_coords, self.coords_2d_front = parse_canonical_obj(self.obj_path)
        self.triangles = get_face_triangles()

        logger.info(
            f"CanonicalFaceModel initialized. Vertices: {len(self.coords_3d)}, "
            f"Triangles: {len(self.triangles)}"
        )

    def get_template_coords(self, mode: str = "canonical_2d") -> np.ndarray:
        """
        指定したモードに応じた 2D 座標 (468, 2) を返す。
        mode: "canonical_2d" (正面顔テンプレート) or "uv_map" (UV展開図)
        """
        if mode == "uv_map":
            return self.uv_coords.copy()
        else:
            return self.coords_2d_front.copy()

    def generate_template_image(
        self,
        width: int = 512,
        height: int = 512,
        mode: str = "canonical_2d",
        bg_color: tuple[int, int, int] = (20, 20, 25),
        line_color: tuple[int, int, int] = (60, 60, 75),
        landmark_color: tuple[int, int, int] = (90, 90, 110),
        draw_landmarks: bool = True,
    ) -> np.ndarray:
        """
        標準顔メッシュのベース背景画像を生成する（BGR）。
        """
        coords_norm = self.get_template_coords(mode=mode)
        pts_px = (coords_norm * np.array([width - 1, height - 1])).astype(np.int32)

        # 背景初期化
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # メッシュ三角形の描画
        for tri in self.triangles:
            p0 = tuple(pts_px[tri[0]])
            p1 = tuple(pts_px[tri[1]])
            p2 = tuple(pts_px[tri[2]])
            cv2.line(img, p0, p1, line_color, 1, cv2.LINE_AA)
            cv2.line(img, p1, p2, line_color, 1, cv2.LINE_AA)
            cv2.line(img, p2, p0, line_color, 1, cv2.LINE_AA)

        # ランドマーク点の描画
        if draw_landmarks:
            for p in pts_px:
                cv2.circle(img, tuple(p), 1, landmark_color, -1, cv2.LINE_AA)

        return img
