import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gradio as gr


from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.rlsp import RLSP

def process_demo(input_img, scale, downscale_algo):
    if input_img is None:
        return [None] * 17
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert Image to [1, 3, H, W] for rlsp
    hr_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)
    _, _, h, w = hr_tensor.shape
    
    new_h = h - (h % scale)
    new_w = w - (w % scale)
    if new_h != h or new_w != w:
        hr_tensor = hr_tensor[:, :, :new_h, :new_w]
        
    # Simulate downscaling to low-resolution
    algo_map = {
        "Bicubic": "bicubic",
        "Bilinear": "bilinear",
        "Nearest Neighbor": "nearest",
        "Lanczos": "area"
    }
    mode = algo_map.get(downscale_algo, "bicubic")
    lr_tensor = F.interpolate(hr_tensor, scale_factor=1.0/scale, mode=mode).clamp(0, 1)
    
    # Baseline upscale (Bicubic)
    bicubic_upscaled = F.interpolate(lr_tensor, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0, 1)
    ideal_residual = hr_tensor - bicubic_upscaled
    
    # Path mappings
    srcnn_path = f"export/srcnn/{scale}x/srcnn_{scale}x_e19.pth"
    
    if scale == 8:
        vdsr_path = "export/vdsr/mix_2x(19e)_8x/vdsr_8x_e13.pth"
    else:
        vdsr_path = "export/vdsr/{scale}x/vdsr_{scale}x_e19.pth"
        
    rlsp_path = "export/rlsp/4x/rlsp_4x_e19.pth" if scale == 4 else ""
    

    def to_pil(tensor):
        if tensor is None:
            return None
        return TF.to_pil_image(tensor.squeeze(0).cpu().clamp(0, 1))
        
    def prepare_visualization(tensor, factor=5.0):
        return (tensor.abs() * factor).clamp(0, 1)

    srcnn_out, srcnn_res = None, None
    vdsr_out, vdsr_res = None, None
    rlsp_out, rlsp_res = None, None

    # --- SRCNN Execution ---
    if os.path.exists(srcnn_path):
        srcnn = SRCNN().to(device)
        srcnn.load_state_dict(torch.load(srcnn_path, map_location=device, weights_only=True))
        srcnn.eval()
        with torch.no_grad():
            srcnn_out = srcnn(bicubic_upscaled)
            srcnn_res = srcnn_out - bicubic_upscaled
            
    # --- VDSR Execution ---
    if os.path.exists(vdsr_path):
        vdsr = VDSR().to(device)
        vdsr.load_state_dict(torch.load(vdsr_path, map_location=device, weights_only=True))
        vdsr.eval()
        with torch.no_grad():
            vdsr_res = vdsr.residual_layer(bicubic_upscaled)
            vdsr_out = bicubic_upscaled + vdsr_res

    # --- RLSP Execution ---
    if scale == 4 and os.path.exists(rlsp_path):
        rlsp = RLSP(scale=4).to(device)
        rlsp.load_state_dict(torch.load(rlsp_path, map_location=device, weights_only=True))
        rlsp.eval()
        with torch.no_grad():
            rlsp_in = lr_tensor.unsqueeze(1)
            rlsp_out = rlsp(rlsp_in).squeeze(1)
            rlsp_res = rlsp_out - bicubic_upscaled

    lr_pil = to_pil(lr_tensor)
    bic_pil = to_pil(bicubic_upscaled)
    bic_res_pil = to_pil(prepare_visualization(ideal_residual))
    srcnn_pil = to_pil(srcnn_out)
    srcnn_res_pil = to_pil(prepare_visualization(srcnn_res)) if srcnn_res is not None else None
    vdsr_pil = to_pil(vdsr_out)
    vdsr_res_pil = to_pil(prepare_visualization(vdsr_res)) if vdsr_res is not None else None
    rlsp_pil = to_pil(rlsp_out)
    rlsp_res_pil = to_pil(prepare_visualization(rlsp_res)) if rlsp_res is not None else None

    return (
        lr_pil,
        # Grid layout views
        bic_pil, bic_res_pil,
        srcnn_pil, srcnn_res_pil,
        vdsr_pil, vdsr_res_pil,
        rlsp_pil, rlsp_res_pil,
        # Tabbed comparison layout views
        bic_pil, bic_res_pil,
        srcnn_pil, srcnn_res_pil,
        vdsr_pil, vdsr_res_pil,
        rlsp_pil, rlsp_res_pil
    )

def toggle_layouts(choice):
    if choice == "Grid View (All on One Page)":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

zoom_js_script = """
(zoom_percent) => {
    const images = document.querySelectorAll('.zoom-container img');
    images.forEach(img => {
        if (img) {
            img.style.width = zoom_percent + '%';
            img.style.maxWidth = 'none';
            img.style.height = 'auto';
        }
    });
}
"""

# Fix of height overflow boxes
custom_css = """
.gradio-container { max-width: 95% !important; }
.zoom-container { 
    max-height: 580px !important; 
    overflow: auto !important; 
    border: 1px solid #44444444; 
    border-radius: 8px;
    padding: 4px;
}
.zoom-container img {
    width: 100%;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Super-Resolution Sandbox")
    
    with gr.Row():
        # ---CONFIGURATION PANEL ---
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload High-Resolution (HR) Ground Truth")
            
            with gr.Row():
                scale_radio = gr.Radio(choices=[2, 4, 8], value=4, label="Scale Factor")
                downscale_algo = gr.Dropdown(
                    choices=["Bicubic", "Bilinear", "Nearest Neighbor", "Lanczos"], 
                    value="Bicubic", 
                    label="Downscale Preprocessing"
                )
                
            layout_mode = gr.Radio(
                choices=["Grid View (All on One Page)", "Close Comparison (Tabbed Sync Zoom)"],
                value="Grid View (All on One Page)",
                label="Application Workspace Layout"
            )
            
            zoom_slider = gr.Slider(
                minimum=100, 
                maximum=600, 
                value=100, 
                step=10, 
                label="Synchronized Zoom & Scroll Level (%)"
            )
            
            run_btn = gr.Button("Execute Super-Resolution", variant="primary")
            lr_preview = gr.Image(label="Simulated Low-Resolution Input", interactive=False)
            
        # --- RIGHT WORKSPACE PANELS ---
        with gr.Column(scale=3):
            
            # WORKSPACE LAYOUT A
            with gr.Column(visible=True) as grid_layout_container:
                gr.Markdown("### Grid View")
                
                with gr.Row():
                    gr.Markdown("#### **Super Resolution Reconstruction**")
                    gr.Markdown("#### **Visualized Error Residual (Error x5)**")
                    
                with gr.Row():
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_bic_img = gr.Image(label="Bicubic Baseline Output", interactive=False, show_label=False)
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_bic_res = gr.Image(label="Bicubic Baseline Residual", interactive=False, show_label=False)
                with gr.Row():
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_srcnn_img = gr.Image(label="SRCNN Output", interactive=False, show_label=False)
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_srcnn_res = gr.Image(label="SRCNN Residual", interactive=False, show_label=False)
                with gr.Row():
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_vdsr_img = gr.Image(label="VDSR Output", interactive=False, show_label=False)
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_vdsr_res = gr.Image(label="VDSR Residual", interactive=False, show_label=False)
                with gr.Row():
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_rlsp_img = gr.Image(label="RLSP Output (4x Only)", interactive=False, show_label=False)
                    with gr.Column(elem_classes=["zoom-container"]):
                        grid_rlsp_res = gr.Image(label="RLSP Residual (4x Only)", interactive=False, show_label=False)

            # WORKSPACE LAYOUT B
            with gr.Column(visible=False) as tabbed_layout_container:
                gr.Markdown("### Close Comparison View")
                
                with gr.Tabs():
                    with gr.TabItem("Bicubic (No Model)"):
                        with gr.Row():
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_bic_img = gr.Image(label="Bicubic Output", image_mode="RGB", interactive=False)
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_bic_res = gr.Image(label="Ideal Residual (Error x5)", image_mode="RGB", interactive=False)
                    
                    with gr.TabItem("SRCNN"):
                        with gr.Row():
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_srcnn_img = gr.Image(label="SRCNN Output", image_mode="RGB", interactive=False)
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_srcnn_res = gr.Image(label="SRCNN Residual (Error x5)", image_mode="RGB", interactive=False)
                            
                    with gr.TabItem("VDSR"):
                        with gr.Row():
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_vdsr_img = gr.Image(label="VDSR Output", image_mode="RGB", interactive=False)
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_vdsr_res = gr.Image(label="VDSR Residual (Error x5)", image_mode="RGB", interactive=False)
                            
                    with gr.TabItem("RLSP (4x Only)"):
                        with gr.Row():
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_rlsp_img = gr.Image(label="RLSP Output", image_mode="RGB", interactive=False)
                            with gr.Column(elem_classes=["zoom-container"]):
                                tab_rlsp_res = gr.Image(label="RLSP Residual (Error x5)", image_mode="RGB", interactive=False)

    # Wire interface workspace visibility toggles
    layout_mode.change(
        fn=toggle_layouts,
        inputs=[layout_mode],
        outputs=[grid_layout_container, tabbed_layout_container]
    )

    # Connect the slider to alter image layout metrics
    zoom_slider.change(
        fn=None,
        inputs=[zoom_slider],
        outputs=[],
        js=zoom_js_script
    )

    # Process models via python
    run_btn.click(
        fn=process_demo,
        inputs=[input_image, scale_radio, downscale_algo],
        outputs=[
            lr_preview,
            # Grid Elements
            grid_bic_img, grid_bic_res,
            grid_srcnn_img, grid_srcnn_res,
            grid_vdsr_img, grid_vdsr_res,
            grid_rlsp_img, grid_rlsp_res,
            # Tabbed Elements
            tab_bic_img, tab_bic_res,
            tab_srcnn_img, tab_srcnn_res,
            tab_vdsr_img, tab_vdsr_res,
            tab_rlsp_img, tab_rlsp_res
        ]
    ).then(
        fn=None,
        inputs=[zoom_slider],
        outputs=[],
        js=zoom_js_script
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)