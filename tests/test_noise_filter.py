"""
tests/test_noise_filter.py
src/noise_filter.py の単体テスト (unittest)
"""

import unittest
import numpy as np
import pandas as pd

from src.noise_filter import (
    apply_refractory_filter,
    apply_spatial_median_filter,
    detect_hot_pixels,
    filter_events_by_hot_pixels,
)


class TestNoiseFilter(unittest.TestCase):
    def test_detect_hot_pixels(self):
        # 10x10 のダミーイベントデータを作成
        # (5, 5) のピクセルに極端に多いイベントを配置
        rows = []
        t = 0.0
        for y in range(10):
            for x in range(10):
                # 通常ピクセル: 10イベント
                for _ in range(10):
                    rows.append({"x": x, "y": y, "polarity": 1, "timestamp_us": t})
                    t += 1000.0

        # ホットピクセル (5, 5): 追加で 2000 イベント
        for _ in range(2000):
            rows.append({"x": 5, "y": 5, "polarity": 1, "timestamp_us": t})
            t += 100.0

        events_df = pd.DataFrame(rows)

        hot_mask, stats = detect_hot_pixels(
            events_df=events_df,
            image_width=10,
            image_height=10,
            ratio_threshold=3.0,
            min_count=50,
            max_rate_hz=500.0,
            border_margin=0,
        )

        self.assertEqual(hot_mask.shape, (10, 10))
        self.assertTrue(hot_mask[5, 5])
        # 周囲の通常ピクセルは False
        self.assertFalse(hot_mask[0, 0])
        self.assertFalse(hot_mask[4, 4])
        self.assertEqual(stats["hot_pixels"], 1)

    def test_filter_events_by_hot_pixels(self):
        rows = [
            {"x": 1, "y": 1, "polarity": 1, "timestamp_us": 10.0},
            {"x": 5, "y": 5, "polarity": 1, "timestamp_us": 20.0},  # hot
            {"x": 2, "y": 2, "polarity": 0, "timestamp_us": 30.0},
        ]
        df = pd.DataFrame(rows)
        hot_mask = np.zeros((10, 10), dtype=bool)
        hot_mask[5, 5] = True

        filtered_df, removed_count, removed_pct = filter_events_by_hot_pixels(df, hot_mask)
        self.assertEqual(len(filtered_df), 2)
        self.assertEqual(removed_count, 1)
        self.assertNotIn(5, filtered_df["x"].values)

    def test_apply_refractory_filter(self):
        # 同一ピクセル (2, 2) で 200us 間隔のイベント群 (refractory 500us 未満)
        rows = [
            {"x": 2, "y": 2, "polarity": 1, "timestamp_us": 1000.0},  # keep
            {"x": 2, "y": 2, "polarity": 1, "timestamp_us": 1200.0},  # drop (< 500us)
            {"x": 2, "y": 2, "polarity": 1, "timestamp_us": 1400.0},  # drop (< 500us)
            {"x": 2, "y": 2, "polarity": 1, "timestamp_us": 1700.0},  # keep (1700 - 1000 = 700us >= 500us)
            {"x": 3, "y": 3, "polarity": 1, "timestamp_us": 1100.0},  # keep (別ピクセル)
        ]
        df = pd.DataFrame(rows).sort_values("timestamp_us").reset_index(drop=True)

        filtered_df, removed, _ = apply_refractory_filter(
            events_df=df,
            refractory_period_us=500.0,
            image_width=10,
            image_height=10,
        )

        self.assertEqual(removed, 2)
        self.assertEqual(len(filtered_df), 3)
        t_vals = filtered_df["timestamp_us"].tolist()
        self.assertIn(1000.0, t_vals)
        self.assertIn(1100.0, t_vals)
        self.assertIn(1700.0, t_vals)
        self.assertNotIn(1200.0, t_vals)
        self.assertNotIn(1400.0, t_vals)

    def test_apply_spatial_median_filter(self):
        arr = np.zeros((7, 7), dtype=np.float32)
        # 単一スパイク
        arr[3, 3] = 100.0

        filtered = apply_spatial_median_filter(arr, ksize=3)
        self.assertEqual(filtered[3, 3], 0.0)

    def test_normalize_heatmap_adaptive(self):
        from src.visualizer import normalize_heatmap

        # 静止時（微小イベントのみ: 最大 0.2）
        quiet_map = np.zeros((10, 10), dtype=np.float32)
        quiet_map[2, 2] = 0.2
        quiet_map[3, 3] = 0.1

        # min_vmax_floor=1.5 により、静止時に 1.0 (赤) にならないこと
        norm_quiet, vmax_quiet = normalize_heatmap(
            quiet_map, mode="percentile", percentile_val=98.5, scale_type="sqrt", min_vmax_floor=1.5
        )
        self.assertEqual(vmax_quiet, 1.5)
        # sqrt(0.2 / 1.5) = ~0.365 (0.8以上の赤にはならない)
        self.assertTrue(np.all(norm_quiet < 0.5))

        # 笑顔時（激しい動き: 最大 4.0）
        active_map = np.zeros((10, 10), dtype=np.float32)
        active_map[5, 5] = 4.0
        active_map[6, 6] = 1.0  # 中程度の動き (ほうれい線等)

        norm_active, vmax_active = normalize_heatmap(
            active_map, mode="percentile", percentile_val=98.5, scale_type="sqrt", min_vmax_floor=1.5
        )
        # 激しい動きは 1.0 (赤) に到達
        self.assertEqual(norm_active[5, 5], 1.0)
        # 中程度の動き (1.0) も sqrt スケールにより sqrt(1.0 / 4.0) = 0.5 となり綺麗に色づく
        self.assertGreater(norm_active[6, 6], 0.4)


if __name__ == "__main__":
    unittest.main()
