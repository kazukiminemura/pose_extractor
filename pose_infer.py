import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

# Ultralytics helpers for OpenVINO-direct path
try:
    from ultralytics.nn.autobackend import AutoBackend  # type: ignore
    from ultralytics.data.augment import LetterBox  # type: ignore
    from ultralytics.utils import nms, ops  # type: ignore
    from ultralytics.engine.results import Results  # type: ignore
except Exception:
    # Optional; only required if using OpenVINO-direct fallback
    AutoBackend = None  # type: ignore


def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception as e:
        print("[ERROR] ultralytics が見つかりません。'pip install ultralytics opencv-python' を実行してください。", file=sys.stderr)
        raise e


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8-Pose で動画を解析して描画 (Ultralytics / OpenVINO)"
    )
    p.add_argument("--source", type=str, required=True, help="入力: 動画パス or カメラID(例: 0)")
    p.add_argument(
        "--engine",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "openvino"],
        help="推論エンジンを選択"
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="モデルパス (.pt もしくは OpenVINO の .xml)"
    )
    p.add_argument("--device", type=str, default="cpu", help="デバイス: cpu / cuda:0 / AUTO / GPU など")
    p.add_argument("--imgsz", type=int, default=640, help="推論解像度")
    p.add_argument("--show", action="store_true", help="ウィンドウに表示")
    p.add_argument("--save", type=str, default=None, help="出力動画パス (.mp4 / .avi)")
    p.add_argument("--export_only", action="store_true", help="OpenVINO へのエクスポートのみ実行")
    p.add_argument("--no_annot", action="store_true", help="描画を無効化 (計測のみ)")
    p.add_argument("--max_frames", type=int, default=0, help="処理フレーム上限 (0 で無制限)")
    return p.parse_args()


def is_camera_source(src: str) -> bool:
    # 数字のみならカメラIDとみなす
    try:
        int(src)
        return True
    except ValueError:
        return False


def _fourcc_for_path(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*"XVID")


def ensure_openvino_model(pt_or_xml: str, imgsz: int = 640) -> Path:
    """
    .pt が指定された場合、未変換なら OpenVINO へエクスポート。
    返り値は OpenVINO ディレクトリ or .xml のパス。
    .xml 指定ならそのまま返す。
    """
    src = Path(pt_or_xml)
    if src.suffix.lower() == ".xml":
        return src

    YOLO = _try_import_ultralytics()

    # 既にエクスポート済みかを緩く探索（ディレクトリ or .xml）
    export_dir = src.with_name(f"{src.stem}_openvino_model")
    if export_dir.exists() and export_dir.is_dir():
        # ディレクトリ内に .xml があれば OK（ファイル名はバージョンで変わる可能性あり）
        xmls = list(export_dir.glob("*.xml"))
        if xmls:
            return export_dir
    else:
        # 直近に既に *.xml が存在していないかも一応チェック
        candidate_xml = src.with_suffix("").with_name(f"{src.stem}_openvino_model.xml")
        if candidate_xml.exists():
            return candidate_xml

    print(f"[INFO] OpenVINO にエクスポートします: {src} -> {export_dir}")
    model = YOLO(str(src))
    out = model.export(format="openvino", imgsz=imgsz)
    # Ultralytics は export の返り値にパス（ディレクトリ or .xml）を返す
    out_path = Path(out)
    if out_path.is_dir():
        # ディレクトリ内の .xml を検出できればディレクトリを返す
        if list(out_path.glob("*.xml")):
            return out_path
    elif out_path.suffix.lower() == ".xml":
        return out_path

    # フォールバック: 想定ディレクトリ内の .xml を総当たり
    if export_dir.exists() and export_dir.is_dir():
        if list(export_dir.glob("*.xml")):
            return export_dir

    raise RuntimeError("OpenVINO エクスポートに失敗しました。出力内に .xml が見つかりません。")


def _normalize_ov_device(device: str) -> str:
    d = (device or "").strip().upper()
    # Allow Intel NPU too
    if d in {"CPU", "GPU", "NPU", "AUTO"}:
        return f"intel:{d}"
    # default AUTO
    return "intel:AUTO"


def _predict_openvino_direct(model: "AutoBackend", frame, imgsz: int):
    # Preprocess (match BasePredictor.preprocess)
    im0 = frame
    lb = LetterBox(imgsz, auto=False, stride=getattr(model, "stride", 32))
    im = lb(image=im0)
    if im.shape[-1] == 3:
        im = im[..., ::-1]  # BGR -> RGB
    im = im.transpose(2, 0, 1)  # HWC -> CHW
    im = torch.from_numpy(im.copy())  # contiguous
    im = im[None]  # add batch
    im = im.to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255.0

    # Inference
    preds = model(im)

    # NMS
    preds = nms.non_max_suppression(
        preds,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        nc=len(model.names),
        end2end=getattr(model, "end2end", False),
        rotated=False,
    )

    pred = preds[0]
    if pred.shape[0]:
        # Scale boxes
        ops.scale_boxes(im.shape[2:], pred[:, :4], im0.shape)
        # Keypoints for pose: reshape and scale
        if hasattr(model, "kpt_shape") and pred.shape[1] >= 6 + (model.kpt_shape[0] * model.kpt_shape[1]):
            kpts = pred[:, 6:].view(len(pred), *model.kpt_shape)
            kpts = ops.scale_coords(im.shape[2:], kpts, im0.shape)
        else:
            kpts = None
    else:
        kpts = None

    # Build Results for plotting API compatibility
    res = Results(
        orig_img=im0,
        path="",
        names=model.names,
        boxes=pred[:, :6] if pred.shape[0] else None,
        keypoints=kpts if kpts is not None and pred.shape[0] else None,
    )
    return res


def load_model(engine: str, model_path: str, imgsz: int, device: str):
    YOLO = _try_import_ultralytics()
    if engine == "ultralytics":
        model = YOLO(model_path)
        backend = "Ultralytics"
        return model, backend

    # OpenVINO: use AutoBackend directly to allow Intel GPU selection
    xml_or_dir = ensure_openvino_model(model_path, imgsz=imgsz)
    if AutoBackend is None:
        raise RuntimeError("ultralytics の内部モジュール読み込みに失敗しました。")
    ov_device = _normalize_ov_device(device)
    model = AutoBackend(str(xml_or_dir), device=ov_device, fp16=False, fuse=True, verbose=True)
    backend = "OpenVINO"
    return model, backend


def main():
    args = parse_args()

    if args.engine == "openvino" and args.export_only:
        # エクスポートのみ
        ensure_openvino_model(args.model, imgsz=args.imgsz)
        print("[DONE] OpenVINO へのエクスポート完了")
        return

    # 入力
    src = args.source
    cap_source = int(src) if is_camera_source(src) else src
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] 入力を開けませんでした: {src}")
        sys.exit(1)

    # 入出力情報
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = _fourcc_for_path(args.save)
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[WARN] 出力ファイルを開けませんでした: {args.save}")
            writer = None

    # モデルロード
    model, backend = load_model(args.engine, args.model, imgsz=args.imgsz, device=args.device)

    frame_idx = 0
    t_prev = time.time()
    smoothed_fps = fps

    print(f"[INFO] 開始: {backend} / model={args.model} / device={args.device}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        if args.engine == "openvino":
            # OpenVINO-direct path
            result = _predict_openvino_direct(model, frame, imgsz=args.imgsz)
            annotated = result.plot() if not args.no_annot else frame
        else:
            # Ultralytics default path
            results = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
            result = results[0]
            annotated = result.plot() if not args.no_annot else frame

        # オーバーレイ情報
        dt = time.time() - t0
        inst_fps = 1.0 / max(dt, 1e-6)
        smoothed_fps = smoothed_fps * 0.9 + inst_fps * 0.1
        label = f"{backend} | {args.imgsz}px | {smoothed_fps:5.1f} FPS"
        cv2.rectangle(annotated, (8, 8), (8 + 360, 38), (0, 0, 0), -1)
        cv2.putText(annotated, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow("YOLOv8-Pose", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("[DONE] 推論終了")


if __name__ == "__main__":
    main()
