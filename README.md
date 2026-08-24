# FaceEventHeatmap

顔面上のイベント発生密度（どの部位にどの程度のイベントが発生しているか）を
標準顔テンプレート（Canonical 2D / UV空間）およびRGB顔画像上に累積し、
ヒートマップ画像および動画として可視化するシステムです。

---

## 主な特徴

- **Canonical 2D / UV 空間へのマッピング**:
  - MediaPipe Face Mesh の 468 頂点とテッセレーション（三角形メッシュ）を利用し、被験者の顔の向き・傾きや表情変化をキャンセルして「標準正面顔テンプレート」上にイベントを累積。
  - 顔の解剖学的部位（口角、小鼻、目元、眉間など）のどこで微細な動き・輝度変化が生じたかを客観的に比較・分析可能。
- **時間窓 $[t - M, t]$ (ms) の柔軟な指定**:
  - 任意の時間窓幅 $M$（例: 50ms, 100ms, 200ms）で過去のイベントを蓄積してフレーム化。
  - RGBタイムスタンプ（`sync_log.csv`）に完全同期して時系列動画を生成。
- **リッチな可視化と解析**:
  - ガウシアンブラー平滑化、パーセンタイル正規化（ホットピクセル耐性）、多彩なカラーマップ（`turbo`, `viridis`, `jet`, `inferno` 等）。
  - カラーバー（凡例）、タイムスタンプ・イベント数オーバーレイ表示。
  - 連番 PNG 画像、MP4 動画、シーケンス全体の総累積ヒートマップ（`total_uv_heatmap.png`）を出力。

---

## ディレクトリ構成

```text
FaceEventHeatmap/
├── config.yaml               # 設定ファイル
├── main.py                   # 実行メインスクリプト
├── requirements.txt          # 依存パッケージ
├── README.md                 # ドキュメント
├── assets/                   # 標準顔モデル・テンプレートデータ
│   ├── canonical_face_model.obj
│   └── canonical_face_template.png
├── src/
│   ├── __init__.py
│   ├── canonical_face.py     # Canonical Face Model 管理
│   ├── data_loader.py        # 各種 CSV / JSON / 動画読み込み
│   ├── uv_buffer.py          # UV ポジションバッファ生成（ラスタライザ）
│   ├── heatmap_accumulator.py# 時間窓ごとのイベント累積・2Dヒストグラム
│   └── visualizer.py         # 平滑化・正規化・カラーマップ・合成・描画
├── input/                    # 入力データ
│   ├── events.csv
│   ├── sync_log.csv
│   ├── sync_params.json
│   ├── landmark.csv
│   ├── transform_matrix.json
│   ├── calibration.json
│   └── (オプション) rgb_video.mp4 / rgb_images/
└── output/                   # 出力データ
    ├── uv_heatmaps/          # UV空間ヒートマップ連番画像 (frame_*.png)
    ├── rgb_heatmaps/         # (オプション) RGB重畳画像
    ├── face_event_heatmap.mp4# ヒートマップ時系列動画
    └── total_uv_heatmap.png  # シーケンス全体の総累積ヒートマップ
```

---

## セットアップ手順

```bash
cd FaceEventHeatmap

# 仮想環境の作成とアクティベート
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

---

## 実行方法

### 1. 基本的な実行（デフォルト設定）

`input/` ディレクトリに入力ファイルを配置し、実行します。

```bash
python main.py
```

### 2. コマンドライン引数によるカスタマイズ

```bash
# 時間窓 M を 50ms に変更し、カラーマップを viridis に変更
python main.py --window-ms 50 --colormap viridis

# クイックテスト（最初の 30 フレームのみ処理）
python main.py --max-frames 30

# 入力・出力ディレクトリを明示指定
python main.py --input-dir /path/to/input --output-dir /path/to/output
```

---

## 設定ファイル (`config.yaml`) の主な項目

```yaml
# 時間窓設定
timing:
  window_ms: 100.0            # 過去 M ms 間のイベントを集計 (例: 50.0, 100.0, 200.0)
  timing_mode: "sync_log"     # "sync_log" (RGB各フレーム時刻) または "stride" (等間隔)

# 標準顔 (Canonical) テンプレート設定
canonical_space:
  width: 512                 # ヒートマップ解像度 (幅)
  height: 512                # ヒートマップ解像度 (高さ)
  template_mode: "canonical_2d" # "canonical_2d" (正面顔) または "uv_map" (UV展開図)

# 可視化設定
visualization:
  colormap: "turbo"          # turbo, viridis, jet, inferno, plasma, magma
  blur_sigma: 3.0            # ガウシアンブラーのシグマ (0でブラー無効)
  alpha: 0.70                # 背景との合成アルファ強度
  draw_colorbar: true        # カラーバーの描画
  draw_info_overlay: true    # タイムスタンプ・イベント数の表示

# 正規化設定
normalization:
  mode: "percentile"         # "percentile" (ホットピクセル除外), "max", "fixed"
  percentile_val: 99.0       # 上位パーセンタイル値
  min_threshold: 0.05        # 最小表示閾値
```
