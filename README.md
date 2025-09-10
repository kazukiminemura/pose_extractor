# YOLOv8-Pose 動画解析 (Ultralytics / OpenVINO)

YOLOv8-Pose を使って動画・カメラ映像から姿勢推定を行い、結果を動画上に描画します。
Ultralytics のオリジナル実行と、OpenVINO エンジンによる高速推論の両方に対応しています。

## セットアップ

Windows / Python 3.9+ を想定。

1) 依存パッケージをインストール

```
pip install --upgrade pip
pip install ultralytics openvino opencv-python
```

CUDA(GPU)で Ultralytics を使う場合は PyTorch の CUDA 対応版が必要です（任意）。

```
# 例: CUDA 12.1 の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2) モデル

- 既定は `yolov8n-pose.pt`（初回は自動ダウンロード）。
- ネットワークが使えない場合は、事前に `.pt` または OpenVINO の `.xml` を用意し、`--model` で指定してください。

## 使い方

Ultralytics で推論（CPU 例）

```
python pose_infer.py --engine ultralytics --model yolov8n-pose.pt \
  --source path/to/input.mp4 --show --save out.mp4 --device cpu
```

Ultralytics で推論（CUDA 例）

```
python pose_infer.py --engine ultralytics --model yolov8n-pose.pt \
  --source 0 --show --device cuda:0
```

OpenVINO で推論（自動エクスポート）

```
python pose_infer.py --engine openvino --model yolov8n-pose.pt \
  --source path/to/input.mp4 --show --save out.mp4 --device AUTO
```

OpenVINO へ事前に変換のみ行う場合

```
python pose_infer.py --engine openvino --model yolov8n-pose.pt --export_only
```

OpenVINO の xml を直接指定

```
python pose_infer.py --engine openvino --model yolov8n-pose_openvino_model/openvino_model.xml \
  --source 0 --show --device CPU
```

主な引数

- `--source`: 入力（動画パス or カメラID 例: `0`）
- `--engine`: `ultralytics` or `openvino`
- `--model`: `.pt` または OpenVINO `.xml`
- `--device`: `cpu`/`cuda:0`/`AUTO`/`GPU` など
- `--imgsz`: 推論解像度（既定 640）
- `--show`: 画面表示
- `--save`: 出力動画パス（例: `out.mp4`）
- `--export_only`: OpenVINO への変換のみ実行
- `--no_annot`: 描画をオフにして計測のみ
- `--max_frames`: 処理フレーム数の上限（0=無制限）

## 備考

- OpenVINO 実行時の `--device` は `AUTO`/`CPU`/`GPU` などが利用可能です（環境に応じて選択）。
- 出力 `--save` を指定しない場合は保存しません。指定時は `.mp4` または `.avi` を推奨。
- 既定の描画は Ultralytics の `result.plot()` を使用しています。カスタム描画に変更も可能です。

