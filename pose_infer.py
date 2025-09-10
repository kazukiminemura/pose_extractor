import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

# Ultralytics helpers (optional)
try:
    from ultralytics.nn.autobackend import AutoBackend  # type: ignore
    from ultralytics.data.augment import LetterBox  # type: ignore
    from ultralytics.utils import nms, ops  # type: ignore
    from ultralytics.engine.results import Results  # type: ignore
except Exception:
    AutoBackend = None  # type: ignore
    LetterBox = None  # type: ignore
    nms = None  # type: ignore
    ops = None  # type: ignore
    Results = None  # type: ignore

try:
    import openvino as ov  # modern API
except Exception:
    ov = None  # type: ignore


def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception as e:
        print("[ERROR] ultralytics が見つかりません。'pip install ultralytics opencv-python' を実行してください", file=sys.stderr)
        raise e


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8-Pose 推論 (Ultralytics / OpenVINO)")
    p.add_argument("--source", type=str, required=True, help="入力: 動画パス or カメラID(例: 0)")
    p.add_argument(
        "--engine",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "openvino", "ov"],
        help="推論エンジン: 'ultralytics' | 'openvino'(Ultralytics-OV) | 'ov'(OpenVINO Runtime)",
    )
    p.add_argument("--model", type=str, default="yolov8n-pose.pt", help="モデルファイル (.pt or .xml / OpenVINO ディレクトリ)")
    p.add_argument("--device", type=str, default="cpu", help="デバイス: cpu / cuda:0 / AUTO / GPU など")
    p.add_argument("--imgsz", type=int, default=640, help="推論解像度")
    p.add_argument("--show", action="store_true", help="ウィンドウ表示")
    p.add_argument("--save", type=str, default=None, help="出力動画パス (.mp4 / .avi)")
    p.add_argument("--export_only", action="store_true", help="OpenVINO へのエクスポートのみ実行")
    p.add_argument("--no_annot", action="store_true", help="描画を無効 (計測のみ)")
    p.add_argument("--max_frames", type=int, default=0, help="処理フレーム上限 (0 で無制限)")
    return p.parse_args()


def is_camera_source(src: str) -> bool:
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
    .pt が指定された場合、未変換なら OpenVINO へエクスポートして返す。
    返り値は OpenVINO ディレクトリ or .xml のパス。
    .xml 指定ならそのまま返す。
    """
    src = Path(pt_or_xml)
    if src.suffix.lower() == ".xml":
        return src

    YOLO = _try_import_ultralytics()

    # 既にエクスポート済みの可能性
    export_dir = src.with_name(f"{src.stem}_openvino_model")
    if export_dir.exists() and export_dir.is_dir():
        xmls = list(export_dir.glob("*.xml"))
        if xmls:
            return export_dir
    else:
        candidate_xml = src.with_suffix("").with_name(f"{src.stem}_openvino_model.xml")
        if candidate_xml.exists():
            return candidate_xml

    print(f"[INFO] OpenVINO にエクスポートします: {src} -> {export_dir}")
    model = YOLO(str(src))
    out = model.export(format="openvino", imgsz=imgsz)
    out_path = Path(out)
    if out_path.is_dir():
        if list(out_path.glob("*.xml")):
            return out_path
    elif out_path.suffix.lower() == ".xml":
        return out_path
    if export_dir.exists() and export_dir.is_dir() and list(export_dir.glob("*.xml")):
        return export_dir
    raise RuntimeError("OpenVINO エクスポートに失敗しました。出力先に .xml が見つかりません")


def _normalize_ov_device_ultra(device: str) -> str:
    d = (device or "").strip().upper()
    if d in {"CPU", "GPU", "NPU", "AUTO"}:
        return f"intel:{d}"
    return "intel:AUTO"


def _normalize_ov_device_rt(device: str) -> str:
    d = (device or "").strip().upper()
    if any(x in d for x in ("CUDA", "CUDA:0", "CUDA:1")):
        return "AUTO"
    if d.startswith("INTEL:"):
        d = d.split(":", 1)[-1]
    if d in {"CPU", "GPU", "AUTO", "NPU"}:
        return d
    if d.startswith("GPU"):
        return "GPU"
    return "AUTO"


def _first_xml_path(p: Path) -> Path:
    if p.is_file() and p.suffix.lower() == ".xml":
        return p
    if p.is_dir():
        xs = sorted(p.glob("*.xml"))
        if xs:
            return xs[0]
    raise FileNotFoundError(f"OpenVINO XML が見つかりません: {p}")


@dataclass
class OVRuntimeModel:
    compiled: Any
    input_shape: Tuple[int, int]
    names: Dict[int, str]
    kpt_shape: Tuple[int, int] = (17, 3)
    fp16: bool = False
    device: str = "AUTO"


def _predict_openvino_direct(model: "AutoBackend", frame, imgsz: int):
    # Preprocess (align with Ultralytics BasePredictor)
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
        ops.scale_boxes(im.shape[2:], pred[:, :4], im0.shape)
        if hasattr(model, "kpt_shape") and pred.shape[1] >= 6 + (model.kpt_shape[0] * model.kpt_shape[1]):
            kpts = pred[:, 6:].view(len(pred), *model.kpt_shape)
            kpts = ops.scale_coords(im.shape[2:], kpts, im0.shape)
        else:
            kpts = None
    else:
        kpts = None

    res = Results(
        orig_img=im0,
        path="",
        names=model.names,
        boxes=pred[:, :6] if pred.shape[0] else None,
        keypoints=kpts if kpts is not None and pred.shape[0] else None,
    )
    return res


def _predict_openvino_rt(model: OVRuntimeModel, frame, imgsz: int):
    """Native OpenVINO Runtime inference with Ultralytics-style postprocess (NMS + KP scaling)."""
    if LetterBox is None or ops is None or Results is None or nms is None:
        raise RuntimeError("Ultralytics の補助モジュールが必要です (LetterBox/ops/nms/Results)")

    im0 = frame
    lb = LetterBox(imgsz, auto=False, stride=32)
    im = lb(image=im0)
    if im.shape[-1] == 3:
        im = im[..., ::-1]  # BGR -> RGB
    im = im.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
    im = np.expand_dims(im, 0)
    im /= 255.0

    # Inference (OpenVINO Runtime)
    outputs = model.compiled([im])  # type: ignore
    if isinstance(outputs, (list, tuple)):
        out = outputs[0]
    else:
        try:
            out = next(iter(outputs.values()))  # type: ignore[attr-defined]
        except Exception:
            out = outputs

    # out is expected as (B, no, N) e.g., (1, 56, 8400)
    if out.ndim == 2:  # add batch dim if squeezed
        out = np.expand_dims(out, 0)
    if out.ndim != 3:
        raise RuntimeError(f"OpenVINO 出力の形状が想定外です: {out.shape}")

    preds = torch.from_numpy(out)  # (B, no, N)
    preds = nms.non_max_suppression(
        preds,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        nc=len(model.names),
        end2end=False,
        rotated=False,
    )

    pred = preds[0]
    if pred.shape[0]:
        ops.scale_boxes((imgsz, imgsz), pred[:, :4], im0.shape)
        total = pred.shape[1]
        expected = 6 + (model.kpt_shape[0] * model.kpt_shape[1])
        if total >= expected:
            kpts = pred[:, 6:6 + model.kpt_shape[0] * model.kpt_shape[1]].view(len(pred), *model.kpt_shape)
            kpts = ops.scale_coords((imgsz, imgsz), kpts, im0.shape)
        else:
            kpts = None
    else:
        kpts = None

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

    xml_or_dir = ensure_openvino_model(model_path, imgsz=imgsz)

    if engine == "openvino":
        if AutoBackend is None:
            raise RuntimeError("Ultralytics の AutoBackend が利用できません")
        ov_device = _normalize_ov_device_ultra(device)
        model = AutoBackend(str(xml_or_dir), device=ov_device, fp16=False, fuse=True, verbose=True)
        backend = "OpenVINO (Ultralytics)"
        return model, backend

    if engine == "ov":
        if ov is None:
            raise RuntimeError("openvino パッケージが見つかりません。'pip install openvino' を実行してください")
        core = ov.Core()
        xml_path = _first_xml_path(Path(xml_or_dir))
        device_rt = _normalize_ov_device_rt(device)
        compiled = core.compile_model(xml_path.as_posix(), device_rt)
        names = {0: 'person'}  # YOLOv8 Pose は 1 クラス(person)
        model = OVRuntimeModel(compiled=compiled, input_shape=(imgsz, imgsz), names=names, kpt_shape=(17, 3), fp16=False, device=device_rt)
        backend = "OpenVINO Runtime"
        return model, backend

    raise ValueError(f"未知のエンジン指定です: {engine}")


def main():
    args = parse_args()

    if args.engine in ("openvino", "ov") and args.export_only:
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
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

    # モデルの読み込み
    model, backend = load_model(args.engine, args.model, imgsz=args.imgsz, device=args.device)

    frame_idx = 0
    smoothed_fps = fps

    print(f"[INFO] Start: {backend} / model={args.model} / device={args.device}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        if args.no_annot:
            annotated = frame

        if args.engine == "openvino":
            result = _predict_openvino_direct(model, frame, imgsz=args.imgsz)
            annotated = result.plot() if not args.no_annot else frame
        elif args.engine == "ov":
            result = _predict_openvino_rt(model, frame, imgsz=args.imgsz)
            annotated = result.plot() if not args.no_annot else frame
        else:
            # Ultralytics default path
            results = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
            result = results[0]
            annotated = result.plot() if not args.no_annot else frame

        # Overlay
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

