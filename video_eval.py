import argparse
import time
import cv2
import torch
import piq
import numpy as np
from pathlib import Path

from models.srcnn import SRCNN
from models.vdsr import VDSR

def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    frame_rgb = tensor.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def load_model(model_name: str, model_path: str, device: torch.device):
    model = SRCNN().to(device) if model_name == "srcnn" else VDSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def process_item(model, lr_path: Path, gt_path: Path, out_path: Path, scale: int, device: torch.device):
    """Processes either a video file (.mp4/etc) OR a directory of PNGs."""
    is_video = lr_path.is_file()
    
    if is_video:
        cap = cv2.VideoCapture(str(lr_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        lr_frames = sorted(list(lr_path.glob("*.png")))
        total_frames = len(lr_frames)
        if total_frames == 0: return None
        first = cv2.imread(str(lr_frames[0]))
        h, w = first.shape[:2]
        fps = 30.0

    out_w, out_h = w * scale, h * scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # If testing, look for GT frames if gt_path is a directory
    gt_frames = sorted(list(gt_path.glob("*.png"))) if gt_path and gt_path.is_dir() else []

    stats = {"psnr": [], "ssim": [], "times": []}

    for i in range(total_frames):
        if is_video:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = cv2.imread(str(lr_frames[i]))
        
        lr_tensor = frame_to_tensor(frame, device)
        lr_upscaled = torch.nn.functional.interpolate(lr_tensor, size=(out_h, out_w), mode="bicubic", align_corners=False).clamp(0, 1)

        # Timing with synchronization (fix for the 1700 FPS bug)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            hr_out = model(lr_upscaled).clamp(0, 1)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        stats["times"].append(time.perf_counter() - start)

        # Metrics
        if gt_frames and i < len(gt_frames):
            gt_img = cv2.imread(str(gt_frames[i]))
            gt_t = frame_to_tensor(gt_img, device)
            if gt_t.shape != hr_out.shape:
                gt_t = torch.nn.functional.interpolate(gt_t, size=(out_h, out_w), mode="bicubic")
            stats["psnr"].append(piq.psnr(hr_out, gt_t, data_range=1.0).item())
            stats["ssim"].append(piq.ssim(hr_out, gt_t, data_range=1.0).item())

        out_video.write(tensor_to_frame(hr_out))

    if is_video: cap.release()
    out_video.release()
    return stats

def process_item(model, lr_path: Path, gt_path: Path, out_path: Path, scale: int, device: torch.device):
    """Processes either a video file (.mp4/etc) OR a directory of PNGs."""
    is_video = lr_path.is_file()
    
    if is_video:
        cap = cv2.VideoCapture(str(lr_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        lr_frames = sorted(list(lr_path.glob("*.png")))
        total_frames = len(lr_frames)
        if total_frames == 0: return None
        first = cv2.imread(str(lr_frames[0]))
        h, w = first.shape[:2]
        fps = 30.0

    out_w, out_h = w * scale, h * scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # Determine if GT is a file or a folder of frames
    gt_frames = []
    if gt_path and gt_path.exists():
        if gt_path.is_dir():
            gt_frames = sorted(list(gt_path.glob("*.png")))
        # Note: If GT is a video file, simple frame matching is harder, 
        # so this script assumes GT is a directory of frames.

    stats = {"psnr": [], "ssim": [], "times": []}

    for i in range(total_frames):
        if is_video:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = cv2.imread(str(lr_frames[i]))
        
        lr_tensor = frame_to_tensor(frame, device)
        lr_upscaled = torch.nn.functional.interpolate(lr_tensor, size=(out_h, out_w), mode="bicubic", align_corners=False).clamp(0, 1)

        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            hr_out = model(lr_upscaled).clamp(0, 1)
        if device.type == 'cuda': torch.cuda.synchronize()
        stats["times"].append(time.perf_counter() - start)

        if gt_frames and i < len(gt_frames):
            gt_t = frame_to_tensor(cv2.imread(str(gt_frames[i])), device)
            if gt_t.shape != hr_out.shape:
                gt_t = torch.nn.functional.interpolate(gt_t, size=(out_h, out_w), mode="bicubic")
            stats["psnr"].append(piq.psnr(hr_out, gt_t, data_range=1.0).item())
            stats["ssim"].append(piq.ssim(hr_out, gt_t, data_range=1.0).item())

        out_video.write(tensor_to_frame(hr_out))

    if is_video: cap.release()
    out_video.release()
    return stats

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.input)
    master_stats = []

    # Check if input is a file (testvideo1.mp4) or a directory (data/video_test)
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

    models_to_run = ["srcnn", "vdsr"] if args.model == "both" else [args.model]

    for model_name in models_to_run:
        w_path = args.srcnn_path if model_name == "srcnn" else args.vdsr_path
        if not Path(w_path).exists(): continue
        model = load_model(model_name, w_path, device)
        
        for label, lr_p, gt_p in tasks:
            out_dir = Path(args.output_dir) / f"{model_name}_{args.scale}x"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{label.replace('/', '_')}.mp4"

            print(f"Processing [{model_name.upper()}]: {label}...")
            res = process_item(model, lr_p, gt_p, out_file, args.scale, device)
            
            if res:
                master_stats.append({
                    "model": model_name.upper(), "label": label,
                    "psnr": np.mean(res["psnr"]) if res["psnr"] else 0,
                    "ssim": np.mean(res["ssim"]) if res["ssim"] else 0,
                    "fps": len(res["times"]) / sum(res["times"])
                })

    # FINAL AGGREGATED OUTPUT
    if master_stats:
        print("\n" + "="*80)
        print(f"{'Model':<8} | {'Task/Variant':<25} | {'PSNR':<10} | {'SSIM':<10} | {'FPS':<8}")
        print("-" * 80)
        for s in master_stats:
            print(f"{s['model']:<8} | {s['label']:<25} | {s['psnr']:<10.4f} | {s['ssim']:<10.4f} | {s['fps']:<8.2f}")
        
        # Calculate Aggregated Totals
        avg_psnr = np.mean([s['psnr'] for s in master_stats if s['psnr'] > 0])
        avg_ssim = np.mean([s['ssim'] for s in master_stats if s['ssim'] > 0])
        avg_fps = np.mean([s['fps'] for s in master_stats])
        
        print("-" * 80)
        print(f"{'TOTAL':<8} | {'AVERAGE AGGREGATED':<25} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {avg_fps:<8.2f}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--model", choices=["srcnn", "vdsr", "both"], default="both")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--srcnn-path", default="export/srcnn/4x/srcnn_4x_e19.pth")
    parser.add_argument("--vdsr-path", default="export/vdsr/4x/vdsr_4x_e19.pth")
    run_evaluation(parser.parse_args())

if __name__ == "__main__":
    main()