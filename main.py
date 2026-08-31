"""
main.py
FaceEventHeatmap - 顔面イベント累積ヒートマップ生成システム

顔面上のどの部位にどの程度イベントが発生しているかを
1. 標準顔テンプレート (Canonical 2D / UV) 空間
2. RGB 顔画像重畳空間
上に時間窓 [t - M, t] (ms) で可視化・動画化する。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.canonical_face import CanonicalFaceModel
from src.data_loader import (
    RGBFrameReader,
    load_calibration,
    load_events,
    load_landmarks,
    load_sync_log,
    load_sync_params,
    load_transform_matrix,
)
from src.heatmap_accumulator import HeatmapAccumulator
from src.noise_filter import (
    apply_refractory_filter,
    apply_spatial_median_filter,
    detect_hot_pixels,
    filter_events_by_hot_pixels,
    save_hot_pixel_report,
    save_hot_pixel_visualization,
)
from src.uv_buffer import build_frame_buffers
from src.visualizer import (
    apply_gaussian_blur,
    blend_heatmap_with_background,
    colorize_heatmap,
    draw_colorbar,
    draw_info_overlay,
    normalize_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceEventHeatmap")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Face Event Heatmap Generator - Canonical UV & RGB Overlay"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--window-ms", type=float, default=None, help="Time window M (ms) [t - M, t]"
    )
    parser.add_argument(
        "--colormap", type=str, default=None, help="Colormap name (turbo, viridis, jet, etc.)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["canonical_2d", "uv_map", "rgb_overlay", "all"],
        default=None,
        help="Visualization mode",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Max frames to process (for quick testing)"
    )
    parser.add_argument("--no-video", action="store_true", help="Disable video generation")
    parser.add_argument("--no-filter", action="store_true", help="Disable noise / hot pixel filtering")
    args = parser.parse_args()

    # 設定ロード
    config = load_config(args.config)

    # CLI 引数の上書き
    if args.input_dir:
        config["input"]["input_dir"] = args.input_dir
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.window_ms is not None:
        config["timing"]["window_ms"] = args.window_ms
    if args.colormap:
        config["visualization"]["colormap"] = args.colormap
    if args.no_video:
        config["output"]["save_video"] = False
    if args.no_filter:
        config.setdefault("filter", {})
        config["filter"].setdefault("hot_pixel", {})["enabled"] = False
        config["filter"].setdefault("refractory", {})["enabled"] = False
        config["filter"].setdefault("spatial", {})["enabled"] = False

    input_dir = config["input"]["input_dir"]
    output_dir = config["output"]["output_dir"]
    window_ms = float(config["timing"]["window_ms"])
    timing_mode = config["timing"].get("timing_mode", "sync_log")
    stride_ms = float(config["timing"].get("stride_ms", 33.33))

    os.makedirs(output_dir, exist_ok=True)
    uv_out_dir = os.path.join(output_dir, "uv_heatmaps")
    rgb_out_dir = os.path.join(output_dir, "rgb_heatmaps")
    os.makedirs(uv_out_dir, exist_ok=True)
    os.makedirs(rgb_out_dir, exist_ok=True)

    cam_w = int(config["camera"]["event_width"])
    cam_h = int(config["camera"]["event_height"])
    rgb_w = int(config["camera"]["rgb_width"])
    rgb_h = int(config["camera"]["rgb_height"])

    # 1. 各種データの読み込み
    logger.info("=== 1. Loading Input Data ===")
    events_path = os.path.join(input_dir, config["input"]["events_file"])
    sync_log_path = os.path.join(input_dir, config["input"]["sync_log_file"])
    sync_params_path = os.path.join(input_dir, config["input"]["sync_params_file"])
    landmark_path = os.path.join(input_dir, config["input"]["landmark_file"])
    transform_path = os.path.join(input_dir, config["input"]["transform_matrix_file"])
    calibration_path = os.path.join(input_dir, config["input"]["calibration_file"])

    events_df = load_events(events_path)
    sync_log_df = load_sync_log(sync_log_path)
    sync_A, sync_B = load_sync_params(sync_params_path)
    landmarks_dict = load_landmarks(landmark_path)
    rvec, tvec = load_transform_matrix(transform_path)
    intrinsics, distortion = load_calibration(calibration_path)

    # 1.5 ノイズ抑制・ホットピクセルフィルタリング
    filter_config = config.get("filter", {})
    hot_cfg = filter_config.get("hot_pixel", {})
    ref_cfg = filter_config.get("refractory", {})
    spatial_cfg = filter_config.get("spatial", {})

    if hot_cfg.get("enabled", True):
        logger.info("=== 1.5 Applying Hot Pixel Filter ===")
        hot_mask, hot_stats = detect_hot_pixels(
            events_df=events_df,
            image_width=cam_w,
            image_height=cam_h,
            ratio_threshold=float(hot_cfg.get("ratio_threshold", 3.5)),
            min_count=int(hot_cfg.get("min_count", 500)),
            max_rate_hz=float(hot_cfg.get("max_rate_hz", 800.0)),
            border_margin=int(hot_cfg.get("border_margin", 2)),
        )

        if hot_cfg.get("save_mask_image", True):
            mask_out_path = os.path.join(output_dir, "hot_pixels_mask.png")
            save_hot_pixel_visualization(
                hot_mask=hot_mask,
                event_counts_2d=hot_stats["event_counts_2d"],
                output_path=mask_out_path,
            )

        if hot_cfg.get("save_report_csv", True):
            report_out_path = os.path.join(output_dir, "hot_pixels_report.csv")
            save_hot_pixel_report(
                hot_mask=hot_mask,
                stats=hot_stats,
                output_path=report_out_path,
            )

        events_df, _, _ = filter_events_by_hot_pixels(events_df, hot_mask)

    if ref_cfg.get("enabled", True):
        logger.info("=== 1.6 Applying Refractory Period Filter ===")
        events_df, _, _ = apply_refractory_filter(
            events_df=events_df,
            refractory_period_us=float(ref_cfg.get("period_us", 500.0)),
            image_width=cam_w,
            image_height=cam_h,
        )

    # RGB フレームリーダー (オプショナル)
    rgb_source = config["input"].get("rgb_source")
    if not rgb_source:
        # デフォルトで input_dir 内の mp4 または rgb_images を自動探索
        for cand in ["video.mp4", "rgb_video.mp4", "input.mp4", "rgb_images"]:
            p = os.path.join(input_dir, cand)
            if os.path.exists(p):
                rgb_source = p
                break
    rgb_reader = RGBFrameReader(rgb_source) if rgb_source else None

    # 2. Canonical Face Model の初期化
    logger.info("=== 2. Initializing Canonical Face Model ===")
    canonical_model = CanonicalFaceModel(asset_dir="assets")

    tmpl_width = int(config["canonical_space"]["width"])
    tmpl_height = int(config["canonical_space"]["height"])
    tmpl_mode = config["canonical_space"].get("template_mode", "canonical_2d")

    # テンプレート背景画像の生成
    bg_img_bgr = canonical_model.generate_template_image(
        width=tmpl_width,
        height=tmpl_height,
        mode=tmpl_mode,
        bg_color=tuple(config["canonical_space"].get("bg_color", [20, 20, 25])),
        line_color=tuple(config["canonical_space"].get("mesh_line_color", [60, 60, 75])),
        landmark_color=tuple(config["canonical_space"].get("landmark_color", [90, 90, 110])),
        draw_landmarks=config["canonical_space"].get("draw_landmarks", False),
    )

    # テンプレート 2D 正規化座標の取得
    template_coords_norm = canonical_model.get_template_coords(mode=tmpl_mode)
    triangles = canonical_model.triangles

    # 3. HeatmapAccumulator の初期化
    logger.info("=== 3. Initializing Heatmap Accumulator ===")
    accumulator = HeatmapAccumulator(
        events_df=events_df,
        sync_A=sync_A,
        sync_B=sync_B,
        uv_width=tmpl_width,
        uv_height=tmpl_height,
        rgb_width=rgb_w,
        rgb_height=rgb_h,
    )

    # 4. フレーム生成時刻リストの構築
    if timing_mode == "sync_log":
        frame_timestamps = [
            (int(row["frame_index"]), float(row["timestamp_ms"]))
            for _, row in sync_log_df.iterrows()
        ]
    else:
        # stride モード
        t_start = accumulator.timestamps_rgb_ms[0]
        t_end = accumulator.timestamps_rgb_ms[-1]
        t_cur = t_start + window_ms
        frame_timestamps = []
        idx = 1
        while t_cur <= t_end:
            frame_timestamps.append((idx, t_cur))
            t_cur += stride_ms
            idx += 1

    if args.max_frames and args.max_frames > 0:
        frame_timestamps = frame_timestamps[: args.max_frames]

    logger.info(f"Total output frames to process: {len(frame_timestamps)}")

    # 5. 動画出力ライターの準備
    save_video = config["output"].get("save_video", True)
    video_writer_uv: Optional[cv2.VideoWriter] = None
    video_writer_rgb: Optional[cv2.VideoWriter] = None
    video_fps = float(config["output"].get("video_fps", 30))

    if save_video:
        video_path_uv = os.path.join(output_dir, config["output"].get("video_filename", "face_event_heatmap.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer_uv = cv2.VideoWriter(video_path_uv, fourcc, video_fps, (tmpl_width, tmpl_height))

        if config["modes"].get("rgb_overlay", False) and rgb_reader:
            video_path_rgb = os.path.join(output_dir, "rgb_overlay_heatmap.mp4")
            video_writer_rgb = cv2.VideoWriter(video_path_rgb, fourcc, video_fps, (rgb_w, rgb_h))

    # 可視化パラメータ
    cmap_name = config["visualization"].get("colormap", "turbo")
    blur_sigma = float(config["visualization"].get("blur_sigma", 3.0))
    alpha_val = float(config["visualization"].get("alpha", 0.70))
    draw_cb = config["visualization"].get("draw_colorbar", True)
    draw_info = config["visualization"].get("draw_info_overlay", True)

    do_spatial_filter = spatial_cfg.get("enabled", True)
    spatial_ksize = int(spatial_cfg.get("median_ksize", 3))

    norm_mode = config["normalization"].get("mode", "percentile")
    percentile_val = float(config["normalization"].get("percentile_val", 98.5))
    min_thresh = float(config["normalization"].get("min_threshold", 0.05))
    fixed_vmax = float(config["normalization"].get("fixed_vmax", 50.0))
    scale_type = config["normalization"].get("scale_type", "sqrt")
    min_vmax_floor = float(config["normalization"].get("min_vmax_floor", 1.5))

    # ランドマークフレームの参照用リスト
    available_lm_frames = sorted(landmarks_dict.keys())
    if not available_lm_frames:
        logger.error("No landmarks found in landmark.csv!")
        sys.exit(1)

    # 直近のバッファキャッシュ
    cached_buffer_frame_idx = -1
    cached_uv_buffer: Optional[np.ndarray] = None
    cached_rgb_buffer: Optional[np.ndarray] = None

    # 6. メイン累積・レンダリングループ
    logger.info("=== 4. Processing and Rendering Frames ===")
    save_images = config["output"].get("save_images", True)
    do_rgb_overlay = config["modes"].get("rgb_overlay", False)

    for frame_idx, t_ms in tqdm(frame_timestamps, desc="Generating Heatmaps"):
        # 最も近いランドマークフレームを探索
        nearest_lm_idx = min(available_lm_frames, key=lambda x: abs(x - frame_idx))

        # バッファの生成またはキャッシュ利用
        if nearest_lm_idx != cached_buffer_frame_idx:
            lm_pts = landmarks_dict[nearest_lm_idx]
            uv_buf, rgb_buf = build_frame_buffers(
                landmarks=lm_pts,
                template_coords_norm=template_coords_norm,
                rvec=rvec,
                tvec=tvec,
                intrinsics=intrinsics,
                distortion=distortion,
                triangles=triangles,
                image_width=cam_w,
                image_height=cam_h,
                rgb_width=rgb_w,
                rgb_height=rgb_h,
            )
            cached_uv_buffer = uv_buf
            cached_rgb_buffer = rgb_buf
            cached_buffer_frame_idx = nearest_lm_idx

        # 時間窓 [t - window_ms, t] のイベント抽出
        ev_x, ev_y, ev_pol, ev_t = accumulator.get_events_in_window(t_ms, window_ms)
        event_count = len(ev_x)

        # -------------------------------------------------------------
        # A. Canonical UV 空間ヒートマップの生成
        # -------------------------------------------------------------
        uv_counts = accumulator.accumulate_uv_heatmap(
            ev_x=ev_x,
            ev_y=ev_y,
            ev_pol=ev_pol,
            uv_buffer=cached_uv_buffer,
            update_total=True,
        )

        uv_raw = uv_counts["all"]
        if do_spatial_filter and spatial_ksize > 1:
            uv_raw = apply_spatial_median_filter(uv_raw, ksize=spatial_ksize)

        # 密度平滑化
        density_uv = apply_gaussian_blur(uv_raw, sigma=blur_sigma)

        # 正規化 (適応型非線形スケール & 下限クランプ)
        norm_uv, vmax_uv = normalize_heatmap(
            density_map=density_uv,
            mode=norm_mode,
            percentile_val=percentile_val,
            min_threshold=min_thresh,
            fixed_vmax=fixed_vmax,
            scale_type=scale_type,
            min_vmax_floor=min_vmax_floor,
        )

        # カラーマップ適用
        color_uv_bgr, alpha_mask_uv = colorize_heatmap(norm_uv, colormap_name=cmap_name)

        # テンプレート背景と合成
        rendered_uv = blend_heatmap_with_background(
            background_bgr=bg_img_bgr,
            heatmap_bgr=color_uv_bgr,
            alpha_mask=alpha_mask_uv,
            global_alpha=alpha_val,
        )

        # カラーバー描画
        if draw_cb:
            rendered_uv = draw_colorbar(
                rendered_uv,
                vmax=vmax_uv,
                colormap_name=cmap_name,
                label="Events/px",
            )

        # 情報オーバーレイ描画
        if draw_info:
            rendered_uv = draw_info_overlay(
                rendered_uv,
                frame_idx=frame_idx,
                t_ms=t_ms,
                window_ms=window_ms,
                event_count=event_count,
            )

        # 画像保存
        if save_images:
            out_img_path = os.path.join(uv_out_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(out_img_path, rendered_uv)

        # 動画書き込み
        if video_writer_uv is not None:
            video_writer_uv.write(rendered_uv)

        # -------------------------------------------------------------
        # B. RGB 顔画像オーバーレイヒートマップ (オプション)
        # -------------------------------------------------------------
        if do_rgb_overlay:
            rgb_bg = rgb_reader.get_frame(frame_idx) if rgb_reader else None
            if rgb_bg is None:
                # RGB 画像がない場合は黒背景
                rgb_bg = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)

            rgb_counts = accumulator.accumulate_rgb_heatmap(
                ev_x=ev_x,
                ev_y=ev_y,
                ev_pol=ev_pol,
                rgb_pixel_buffer=cached_rgb_buffer,
            )

            rgb_raw = rgb_counts["all"]
            if do_spatial_filter and spatial_ksize > 1:
                rgb_raw = apply_spatial_median_filter(rgb_raw, ksize=spatial_ksize)

            density_rgb = apply_gaussian_blur(rgb_raw, sigma=blur_sigma * 2.0)
            norm_rgb, vmax_rgb = normalize_heatmap(
                density_map=density_rgb,
                mode=norm_mode,
                percentile_val=percentile_val,
                min_threshold=min_thresh,
                scale_type=scale_type,
                min_vmax_floor=min_vmax_floor,
            )
            color_rgb_bgr, alpha_mask_rgb = colorize_heatmap(norm_rgb, colormap_name=cmap_name)
            rendered_rgb = blend_heatmap_with_background(
                background_bgr=rgb_bg,
                heatmap_bgr=color_rgb_bgr,
                alpha_mask=alpha_mask_rgb,
                global_alpha=alpha_val,
            )

            if draw_info:
                rendered_rgb = draw_info_overlay(
                    rendered_rgb,
                    frame_idx=frame_idx,
                    t_ms=t_ms,
                    window_ms=window_ms,
                    event_count=event_count,
                )

            if save_images:
                out_rgb_path = os.path.join(rgb_out_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(out_rgb_path, rendered_rgb)

            if video_writer_rgb is not None:
                video_writer_rgb.write(rendered_rgb)

    # 動画ライター解放
    if video_writer_uv is not None:
        video_writer_uv.release()
        logger.info(f"Saved UV heatmap video to {video_path_uv}")
    if video_writer_rgb is not None:
        video_writer_rgb.release()
        logger.info(f"Saved RGB overlay heatmap video to {video_path_rgb}")

    # 7. 全期間の総累積ヒートマップの生成と保存
    if config["output"].get("save_total_heatmap", True):
        logger.info("=== 5. Rendering Total Accumulated Heatmap ===")
        total_counts = accumulator.total_uv_counts["all"]
        if do_spatial_filter and spatial_ksize > 1:
            total_counts = apply_spatial_median_filter(total_counts, ksize=spatial_ksize)
        total_density = apply_gaussian_blur(total_counts, sigma=blur_sigma * 1.5)
        total_norm, total_vmax = normalize_heatmap(
            density_map=total_density,
            mode="percentile",
            percentile_val=98.0,
            min_threshold=0.01,
            scale_type="sqrt",
            min_vmax_floor=5.0,
        )
        total_color_bgr, total_alpha = colorize_heatmap(total_norm, colormap_name=cmap_name)
        total_rendered = blend_heatmap_with_background(
            background_bgr=bg_img_bgr,
            heatmap_bgr=total_color_bgr,
            alpha_mask=total_alpha,
            global_alpha=0.80,
        )
        if draw_cb:
            total_rendered = draw_colorbar(
                total_rendered,
                vmax=total_vmax,
                colormap_name=cmap_name,
                label="Total Events/px",
            )
        total_out_path = os.path.join(output_dir, "total_uv_heatmap.png")
        cv2.imwrite(total_out_path, total_rendered)
        logger.info(f"Saved total accumulated heatmap to {total_out_path}")

    if rgb_reader:
        rgb_reader.release()

    logger.info("=== Processing Completed Successfully! ===")


if __name__ == "__main__":
    main()
