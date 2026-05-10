import argparse
import time
import cv2
import torch
import piq
import numpy as np
from pathlib import Path

from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.rlsp import RLSP

def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    frame_rgb = tensor.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def load_model(model_name: str, model_path: str, scale: int, device: torch.device):
    if model_name == "srcnn":
        model = SRCNN().to(device)
    elif model_name == "vdsr":
        model = VDSR().to(device)
    elif model_name == "rlsp":
        model = RLSP().to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def process_item(model, model_name: str, lr_path: Path, gt_path: Path, out_path: Path, scale: int, device: torch.device):
    is_video = lr_path.is_file()
    
    frames = []
    if is_video:
        cap = cv2.VideoCapture(str(lr_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
    else:
        lr_frames = sorted(list(lr_path.glob("*.png")))
        if not lr_frames: return None
        for f_path in lr_frames:
            frames.append(cv2.imread(str(f_path)))
        fps = 30.0

    h, w = frames[0].shape[:2]
    out_w, out_h = w * scale, h * scale
    total_frames = len(frames)

    # [B, T, 3, H, W]
    # B = 1, T = total_frames
    lr_list = [frame_to_tensor(f, device) for f in frames]
    lr_tensor_seq = torch.stack(lr_list, dim=1)

    stats = {"psnr": [], "ssim": [], "times": []}

    with torch.no_grad():
        if model_name == "rlsp":
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            
            hr_out_seq = model(lr_tensor_seq)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            stats["times"].append(time.perf_counter() - start)
        else:
            hr_out_list = []
            start = time.perf_counter()
            for i in range(total_frames):
                img_t = lr_tensor_seq[:, i, :, :, :]
                lr_upscaled = torch.nn.functional.interpolate(
                    img_t, size=(out_h, out_w), mode="bicubic", align_corners=False
                )
                hr_out_list.append(model(lr_upscaled))
            
            if device.type == 'cuda': torch.cuda.synchronize()
            stats["times"].append(time.perf_counter() - start)
            hr_out_seq = torch.stack(hr_out_list, dim=1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    gt_frames = sorted(list(gt_path.glob("*.png"))) if gt_path and gt_path.is_dir() else []

    for i in range(total_frames):
        hr_frame_t = hr_out_seq[:, i, :, :, :].clamp(0, 1)
        
        if gt_frames and i < len(gt_frames):
            gt_t = frame_to_tensor(cv2.imread(str(gt_frames[i])), device)
            if gt_t.shape != hr_frame_t.shape:
                gt_t = torch.nn.functional.interpolate(gt_t, size=(out_h, out_w), mode="bicubic")
            stats["psnr"].append(piq.psnr(hr_frame_t, gt_t, data_range=1.0).item())
            stats["ssim"].append(piq.ssim(hr_frame_t, gt_t, data_range=1.0).item())

        out_video.write(tensor_to_frame(hr_frame_t))

    out_video.release()
    return stats

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.input)
    master_stats = []

    # Check if input is a file or a directory
    if root.is_file():
        tasks = [("SingleVideo", root, None)]
    else:
        all_folders = [d for d in root.iterdir() if d.is_dir()]
        gt_root = next((d for d in all_folders if d.name.upper() == "GT"), None)
        variations = [d for d in all_folders if d.name.upper() != "GT"]
        
        tasks = []
        for v in variations:
            sub_seqs = [d for d in v.iterdir() if d.is_dir()]
            if not sub_seqs: tasks.append((v.name, v, gt_root))
            else:
                for s in sub_seqs:
                    tasks.append((f"{v.name}/{s.name}", s, gt_root / s.name if gt_root else None))

    all_models = ["srcnn", "vdsr", "rlsp"]
    models_to_run = all_models if args.model == "all" else [args.model]

    for model_name in models_to_run:
        if model_name == "srcnn": w_path = args.srcnn_path
        elif model_name == "vdsr": w_path = args.vdsr_path
        elif model_name == "rlsp": w_path = args.rlsp_path
        
        if not Path(w_path).exists():
            print(f"Weights not found for {model_name}, skipping...")
            continue

        model = load_model(model_name, w_path, args.scale, device)
        
        for label, lr_p, gt_p in tasks:
            out_dir = Path(args.output_dir) / f"{model_name}_{args.scale}x"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{label.replace('/', '_')}.mp4"

            print(f"Processing [{model_name.upper()}]: {label}...")
            res = process_item(model, model_name, lr_p, gt_p, out_file, args.scale, device)
            
            if res:
                master_stats.append({
                    "model": model_name.upper(), "label": label,
                    "psnr": np.mean(res["psnr"]) if res["psnr"] else 0,
                    "ssim": np.mean(res["ssim"]) if res["ssim"] else 0,
                    "fps": len(res["times"]) / sum(res["times"])
                })

    if master_stats:
        print("\n" + "="*90)
        print(f"{'Model':<8} | {'Task/Variant':<25} | {'PSNR':<10} | {'SSIM':<10} | {'FPS':<8}")
        print("-" * 90)
        for s in master_stats:
            print(f"{s['model']:<8} | {s['label']:<25} | {s['psnr']:<10.4f} | {s['ssim']:<10.4f} | {s['fps']:<8.2f}")
        print("="*90)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--model", choices=["srcnn", "vdsr", "rlsp", "all"], default="all")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--srcnn-path", default="export/srcnn/4x/srcnn_4x_e19.pth")
    parser.add_argument("--vdsr-path", default="export/vdsr/4x/vdsr_4x_e19.pth")
    parser.add_argument("--rlsp-path", default="export/rlsp/4x/rlsp_4x_e19.pth")
    run_evaluation(parser.parse_args())

if __name__ == "__main__":
    main()