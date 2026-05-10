import argparse
import torch
import os
import time
import torch.nn.functional as F
from pathlib import Path
from torchvision.io import decode_image, ImageReadMode
from torchvision.utils import save_image
from models.srcnn import SRCNN
from models.vdsr import VDSR

def get_residual_tensor(model, model_choice, lr_upscaled):
    # Nevoláme model.eval() v každom kroku, nastaví sa raz v main()
    with torch.no_grad():
        if model_choice == "vdsr":
            residual = model.residual_layer(lr_upscaled)
        else:
            output = model(lr_upscaled)
            residual = output - lr_upscaled
    return residual

def calculate_metrics(target_res, predicted_res):
    """Vypočíta numerickú podobnosť medzi ideálnym a predikovaným rezíduom."""
    mse = F.mse_loss(predicted_res, target_res).item()
    cos_sim = F.cosine_similarity(predicted_res.flatten(), target_res.flatten(), dim=0).item()
    return mse, cos_sim

def prepare_visualization(tensor, factor=5.0):
    return (tensor.abs() * factor).clamp(0, 1)

def main():
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Residual Analysis on full dataset")
    parser.add_argument("-i", "--input-gt-dir", type=str, required=True, help="Cesta k priečinku s HR (GT) obrázkami")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--srcnn-path", type=str, required=True)
    parser.add_argument("--vdsr-path", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default="results/residuals/")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Načítanie modelov na GPU a prepnutie do eval režimu
    srcnn = SRCNN().to(device)
    srcnn.load_state_dict(torch.load(args.srcnn_path, map_location=device, weights_only=True))
    srcnn.eval()

    vdsr = VDSR().to(device)
    vdsr.load_state_dict(torch.load(args.vdsr_path, map_location=device, weights_only=True))
    vdsr.eval()

    # Získanie všetkých obrázkov v priečinku
    image_paths = sorted(list(Path(args.input_gt_dir).glob("*.png")))
    if not image_paths:
        print(f"Chyba: V priečinku {args.input_gt_dir} neboli nájdené žiadne .png súbory.")
        return

    total_images = len(image_paths)
    print(f"Začínam analýzu pre {total_images} obrázkov na zariadení: {device}...")

    srcnn_mse_sum, srcnn_cos_sum = 0.0, 0.0
    vdsr_mse_sum, vdsr_cos_sum = 0.0, 0.0

    start_time = time.perf_counter()

    for idx, img_path in enumerate(image_paths):
        # 1. Príprava dát
        gt_image = decode_image(str(img_path), mode=ImageReadMode.RGB).float() / 255.0
        gt_image = gt_image.unsqueeze(0).to(device, non_blocking=True)

        lr_small = F.interpolate(gt_image, scale_factor=(1/args.scale), mode="bicubic", align_corners=False)
        lr_upscaled = F.interpolate(lr_small, size=(gt_image.shape[2], gt_image.shape[3]), mode="bicubic", align_corners=False)

        # 2. Ideálne rezíduum
        ideal_residual = gt_image - lr_upscaled

        # 3. Inferencia a metriky
        srcnn_res = get_residual_tensor(srcnn, "srcnn", lr_upscaled)
        s_mse, s_cos = calculate_metrics(ideal_residual, srcnn_res)
        srcnn_mse_sum += s_mse
        srcnn_cos_sum += s_cos

        vdsr_res = get_residual_tensor(vdsr, "vdsr", lr_upscaled)
        v_mse, v_cos = calculate_metrics(ideal_residual, vdsr_res)
        vdsr_mse_sum += v_mse
        vdsr_cos_sum += v_cos

        # 4. Uloženie obrázkov LEN pre prvý súbor z datasetu (ukážka do reportu)
        if idx == 0:
            print(f"Ukladám vizualizácie pre ukážkový obrázok: {img_path.name}")
            save_image(prepare_visualization(ideal_residual), os.path.join(args.output_dir, "residual_IDEAL.png"))
            save_image(prepare_visualization(srcnn_res), os.path.join(args.output_dir, "residual_SRCNN.png"))
            save_image(prepare_visualization(vdsr_res), os.path.join(args.output_dir, "residual_VDSR.png"))
            save_image(lr_upscaled.squeeze(0), os.path.join(args.output_dir, "input_blurred.png"))

        # Progress bar (každých 10 obrázkov)
        if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
            print(f"Spracované: {idx + 1}/{total_images}...")

    # Výpočet priemerov
    avg_srcnn_mse = srcnn_mse_sum / total_images
    avg_srcnn_cos = srcnn_cos_sum / total_images
    avg_vdsr_mse = vdsr_mse_sum / total_images
    avg_vdsr_cos = vdsr_cos_sum / total_images
    
    elapsed_time = time.perf_counter() - start_time

    # Výpis výsledkov
    print("\n" + "="*50)
    print(f"CELKOVÉ PRIEMERY PRE DATASET ({total_images} obrázkov, Mierka {args.scale}x)")
    print(f"Čas spracovania: {elapsed_time:.2f} sekúnd")
    print("="*50)
    print(f"SRCNN -> Priemerné MSE: {avg_srcnn_mse:.6f}, Priemerná podobnosť: {avg_srcnn_cos:.4f}")
    print(f"VDSR  -> Priemerné MSE: {avg_vdsr_mse:.6f}, Priemerná podobnosť: {avg_vdsr_cos:.4f}")
    print("-" * 50)
    winner = "VDSR" if avg_vdsr_mse < avg_srcnn_mse else "SRCNN"
    print(f"Záver: V rekonštrukcii rezíduí je priemerne presnejší model {winner}")

if __name__ == "__main__":
    main()