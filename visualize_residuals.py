import argparse
import torch
import os
import torch.nn.functional as F
from torchvision.io import decode_image, ImageReadMode
from torchvision.utils import save_image
from models.srcnn import SRCNN
from models.vdsr import VDSR

def get_residual_tensor(model, model_choice, lr_upscaled):
    model.eval()
    with torch.no_grad():
        if model_choice == "vdsr":
            residual = model.residual_layer(lr_upscaled)
        else:
            output = model(lr_upscaled)
            residual = output - lr_upscaled
    return residual

def calculate_metrics(target_res, predicted_res):
    """Vypočíta numerickú podobnosť medzi ideálnym a predikovaným rezíduom."""
    # MSE - čím nižšie, tým lepšie (presnosť pixelov)
    mse = F.mse_loss(predicted_res, target_res).item()
    
    # Cosine Similarity - čím bližšie k 1.0, tým lepšie (podobnosť štruktúry/hrán)
    cos_sim = F.cosine_similarity(predicted_res.flatten(), target_res.flatten(), dim=0).item()
    
    return mse, cos_sim

def prepare_visualization(tensor, factor=5.0):
    return (tensor.abs() * factor).clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="Residual Analysis with Numerical Comparison")
    parser.add_argument("-i", "--input-gt", type=str, required=True)
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--srcnn-path", type=str, required=True)
    parser.add_argument("--vdsr-path", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default="gt_experiment")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Príprava dát
    gt_image = decode_image(args.input_gt, mode=ImageReadMode.RGB).float() / 255.0
    gt_image = gt_image.unsqueeze(0).to(device)

    lr_small = F.interpolate(gt_image, scale_factor=(1/args.scale), mode="bicubic", align_corners=False)
    lr_upscaled = F.interpolate(lr_small, size=(gt_image.shape[2], gt_image.shape[3]), mode="bicubic", align_corners=False)

    # Ideálne rezíduum
    ideal_residual = gt_image - lr_upscaled
    save_image(prepare_visualization(ideal_residual), os.path.join(args.output_dir, "residual_IDEAL.png"))

    # SRCNN analýza
    srcnn = SRCNN().to(device)
    srcnn.load_state_dict(torch.load(args.srcnn_path, map_location=device, weights_only=True))
    srcnn_res = get_residual_tensor(srcnn, "srcnn", lr_upscaled)
    srcnn_mse, srcnn_cos = calculate_metrics(ideal_residual, srcnn_res)
    save_image(prepare_visualization(srcnn_res), os.path.join(args.output_dir, "residual_SRCNN.png"))

    #  VDSR analýza
    vdsr = VDSR().to(device)
    vdsr.load_state_dict(torch.load(args.vdsr_path, map_location=device, weights_only=True))
    vdsr_res = get_residual_tensor(vdsr, "vdsr", lr_upscaled)
    vdsr_mse, vdsr_cos = calculate_metrics(ideal_residual, vdsr_res)
    save_image(prepare_visualization(vdsr_res), os.path.join(args.output_dir, "residual_VDSR.png"))

    # Výpis výsledkov
    print("\n" + "="*30)
    print(f"NUMERICKÉ POROVNANIE REZÍDUÍ (Mierka {args.scale}x)")
    print("="*30)
    print(f"SRCNN -> MSE: {srcnn_mse:.6f}, Podobnosť (Cosine): {srcnn_cos:.4f}")
    print(f"VDSR  -> MSE: {vdsr_mse:.6f}, Podobnosť (Cosine): {vdsr_cos:.4f}")
    print("-" * 30)
    winner = "VDSR" if vdsr_mse < srcnn_mse else "SRCNN"
    print(f"V tomto teste je presnejší model: {winner}")

if __name__ == "__main__":
    main()