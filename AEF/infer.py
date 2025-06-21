import os
import glob
import argparse
import torch

from PIL import Image
import clip
from pathlib import Path
from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import SCBNet, TPBNet
from utils import import_model_class_from_model_name_or_path


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'])
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False)
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--used_clip_vision_layers", type=int, default=24)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--time_threshold", type=int, default=960)
    parser.add_argument("--save_root", default="temp_results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--task", default=None)

    return parser.parse_args(input_args) if input_args else parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model paths
    SCBNet_desnow_path = os.path.join("pre-trained/desnow", "scb") 
    TPBNet_desnow_path = os.path.join("pre-trained/desnow", "tpb.pt")
    SCBNet_derain_path = os.path.join("pre-trained/derain", "scb") 
    TPBNet_derain_path = os.path.join("pre-trained/derain", "tpb.pt")

    # Load models
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    clip_v = CLIPVisionModel.from_pretrained(args.clip_path).to(device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)

    # Load SCB and TPB for desnow
    scb_desnow_net = SCBNet.from_pretrained(SCBNet_desnow_path).to(device).eval()
    tpb_desnow_net = TPBNet().to(device)
    try:
        tpb_desnow_net.load_state_dict(torch.load(TPBNet_desnow_path)['model'], strict=True)
    except:
        tpb_desnow_net = torch.nn.DataParallel(tpb_desnow_net)
        tpb_desnow_net.load_state_dict(torch.load(TPBNet_desnow_path)['model'], strict=True)
    tpb_desnow_net.eval()

    # Load SCB and TPB for derain
    scb_derain_net = SCBNet.from_pretrained(SCBNet_derain_path).to(device).eval()
    tpb_derain_net = TPBNet().to(device)
    try:
        tpb_derain_net.load_state_dict(torch.load(TPBNet_derain_path)['model'], strict=True)
    except:
        tpb_derain_net = torch.nn.DataParallel(tpb_derain_net)
        tpb_derain_net.load_state_dict(torch.load(TPBNet_derain_path)['model'], strict=True)
    tpb_derain_net.eval()

    # CLIP model for classification
    model, processor = clip.load("ViT-B/32", device=device)

    prompt = {
        "Remove snowy effect in the scene": "A snowy photo",
        "Remove rainy effect in the scene.": "A rainy photo"
    }

    load_path = args.image_path
    images = glob.glob(os.path.join(load_path, '*.png'))

    for image_path in images:
        image = load_image(image_path)
        pil_image = image.copy()

        with torch.no_grad():
            if args.task:
                plugin_name = args.task
                scb_net, tpb_net = scb_desnow_net, tpb_desnow_net
            else:
                image_feature = processor(image).unsqueeze(0).to(device)
                text = clip.tokenize(prompt).to(device)
                image_features = model.encode_image(image_feature)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(0)

                if similarities[0] > similarities[1]:

                    plugin_name = "desnow"
                    scb_net, tpb_net = scb_desnow_net, tpb_desnow_net

                else:
                    plugin_name = "derain"
                    scb_net, tpb_net = scb_derain_net, tpb_derain_net

            print(f"✅ Selected plugin: {plugin_name}")

            clip_visual_input = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device)
            clip_outputs = clip_v(clip_visual_input, output_attentions=True, output_hidden_states=True)
            prompt_embeds = tpb_net(clip_vision_outputs=clip_outputs,
                                    use_global=args.used_clip_vision_global,
                                    layer_ids=args.used_clip_vision_layers)

            width, height = image.size
            image = vae_image_processor.preprocess(image, height=height, width=width).to(device)
            scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(image)), 2, dim=1)[0]
            b, c, h, w = scb_cond.size()

            generator = torch.Generator().manual_seed(args.seed)
            noise = torch.randn((1, 4, h, w), generator=generator).to(device)
            latents = noise if args.inp_of_unet_is_random_noise else None

            noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
            timesteps = noise_scheduler.timesteps.long()

            for t in timesteps:
                if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                    latents = noise_scheduler.add_noise(scb_cond, noise, t)

                down_block_res_samples = scb_net(latents, t, encoder_hidden_states=prompt_embeds,
                                                 cond_img=scb_cond, return_dict=False)

                noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds,
                                  down_block_additional_residuals=down_block_res_samples).sample

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred = vae_image_processor.postprocess(pred, output_type='pil')[0]

        load_path = Path(load_path)
        save_dir = str(load_path.parent)+'/processed_images'

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        pred.save(save_path)
        print(f"{save_path}")
        print('---------done-----------')
