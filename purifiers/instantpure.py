from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline, TCDScheduler

from load_dm import get_imagenet_dm_conf
from purifiers.base import BasePurifier


@dataclass
class InstantPureConfig:
    model_name: str = "LCM"
    load_origin_lora: bool = False
    lora_input_dir: Optional[str] = None
    num_inference_step: int = 1
    strength: float = 0.1
    seed: int = 3407
    guidance_scale: float = 1.0
    control_scale: float = 0.8
    diffusion_respace: str = "ddim50"
    diffusion_timestep: int = 150
    controlnet_model_name: str = "lllyasviel/sd-controlnet-canny"
    sd_model_name: str = "runwayml/stable-diffusion-v1-5"


class InstantPurePurifier(BasePurifier):
    name = "instantpure"
    supports_sdedit_attack = True

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.diffusion_model, self.diffusion = get_imagenet_dm_conf(
            device=self.device,
            respace=config.diffusion_respace,
        )
        self.pipe = self._build_pipe()

    def _build_pipe(self):
        model_name = str(self.config.model_name).upper()
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_name,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.sd_model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16",
        ).to(self.device, dtype=torch.float32)

        if model_name == "LCM":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif model_name == "TCD":
            pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
        else:
            raise ValueError("InstantPure purifier model must be one of: LCM, TCD.")

        self._load_lora_weights(pipe)
        return pipe

    def _load_lora_weights(self, pipe):
        model_name = str(self.config.model_name).upper()
        if self.config.load_origin_lora:
            if model_name == "LCM":
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="origin_lora")
                if self.config.lora_input_dir:
                    lora_state_dict = pipe.lora_state_dict(self.config.lora_input_dir)
                    unwrapped_state_dict = {}
                    for peft_key, weight in lora_state_dict[0].items():
                        key = peft_key.replace("base_model.model.", "")
                        unwrapped_state_dict[key] = weight.to(pipe.dtype)
                    pipe.load_lora_weights(unwrapped_state_dict, adapter_name="adv_lora")
                    pipe.set_adapters(["origin_lora", "adv_lora"], adapter_weights=[0.5, 0.5])
            elif model_name == "TCD":
                pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="origin_lora")
            else:
                raise ValueError("Invalid InstantPure purifier model.")
            return

        if not self.config.lora_input_dir:
            return

        lora_state_dict = pipe.lora_state_dict(self.config.lora_input_dir)
        unwrapped_state_dict = {}
        for peft_key, weight in lora_state_dict[0].items():
            key = peft_key.replace("base_model.model.", "")
            unwrapped_state_dict[key] = weight.to(pipe.dtype)
        pipe.load_lora_weights(unwrapped_state_dict)

    def _resolve_diffusion_timestep(self, timestep):
        requested_t = int(timestep)
        if hasattr(self.diffusion, "timestep_map") and self.diffusion.timestep_map:
            valid_steps = list(self.diffusion.timestep_map)
            floor_index = 0
            for index, original_step in enumerate(valid_steps):
                if original_step > requested_t:
                    break
                floor_index = index
            return floor_index
        max_t = len(self.diffusion.sqrt_alphas_cumprod) - 1
        return max(0, min(requested_t, max_t))

    def _build_control_images(self, image_batch):
        control_images = []
        for image in image_batch.detach().cpu():
            pil_image = TF.to_pil_image(image)
            control = np.array(pil_image)
            control = cv2.Canny(control, 100, 200)
            control = control[:, :, None]
            control = np.concatenate([control, control, control], axis=2)
            control_images.append(
                Image.fromarray(control).resize((512, 512), resample=Image.NEAREST)
            )
        return control_images

    def sdedit(self, x, timestep=None, to_01=True):
        resolved_t = self._resolve_diffusion_timestep(
            self.config.diffusion_timestep if timestep is None else timestep
        )
        x = x * 2 - 1
        t_tensor = torch.full((x.shape[0],), resolved_t, dtype=torch.long, device=x.device)
        sample = self.diffusion.q_sample(x, t_tensor)
        indices = list(range(resolved_t + 1))[::-1]
        for i in indices:
            out = self.diffusion.ddim_sample(
                self.diffusion_model,
                sample,
                torch.full((x.shape[0],), i, dtype=torch.long, device=x.device),
            )
            sample = out["sample"]
        if to_01:
            sample = (sample + 1) / 2
        return sample

    def purify(self, x):
        batch = x.shape[0]
        prompt = ["" for _ in range(batch)]
        original_size = x.size(-1)
        image_input = x
        if original_size != 512:
            image_input = F.interpolate(image_input, size=(512, 512), mode="bilinear")

        image_input = torch.clamp(image_input, 0, 1)
        control_images = self._build_control_images(image_input)
        generators = []
        for index in range(batch):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.seed + index)
            generators.append(generator)
        image = self.pipe(
            prompt=prompt,
            image=image_input,
            control_image=control_images,
            num_inference_steps=self.config.num_inference_step,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            controlnet_conditioning_scale=self.config.control_scale,
            generator=generators,
            output_type="pt",
            return_dict=False,
        )
        output = image[0]
        if original_size != 512:
            output = F.interpolate(output, size=(original_size, original_size), mode="bilinear")
        return output
