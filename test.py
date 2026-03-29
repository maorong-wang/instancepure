# modify on top of https://github.com/xavihart/Diff-PGD

import os
import sys
from pathlib import Path

from PIL import Image, ImageFilter
from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL
import matplotlib.pylab as plt
import time
import glob
import pandas as pd
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import Resize, Grayscale, GaussianBlur
# from torchmetrics.image import SSIM, PSNR
# from torcheval.metrics import FrechetInceptionDistance as FID
from diffusers import LCMScheduler, TCDScheduler, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import *
from peft import LoraConfig
from datasets import load_dataset
import argparse
from autoattack import AutoAttack
import cv2
# import lpips
from safetensors.torch import load_file
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
# from dame_recon.purifier import *
# from advex_uar.attacks.snow_attack import *
# from advex_uar.attacks.fog_attack import *
# from advex_uar.attacks.gabor_attack import *
import foolbox as fb

def parse_args():
    parser = argparse.ArgumentParser(description="choose using LCM/TCD and original/adversarial Lora")
    parser.add_argument("--model", required=True, type=str, help="tcd or lcm")
    parser.add_argument("--load_origin_lora", default=False, action="store_true",
                        help="using original/adversarial Lora")
    parser.add_argument("--lora_input_dir", type=str, help="input lora directory")
    parser.add_argument("--output_dir", default="vis_and_stat/", type=str, help="output directory")
    parser.add_argument("--num_validation_set", default=1000, type=int, help="size of subset of validation set")
    parser.add_argument("--num_inference_step", default=1, type=int, help="inference step of diffusion model")
    parser.add_argument("--strength", default=0.1, type=float, help="noise added to the original image")
    parser.add_argument("--seed", default=3407, type=int, help="seed for random number generator")
    parser.add_argument("--guidance_scale", default=1.0, type=float, help="guidance scale of diffusion model")
    parser.add_argument("--control_scale", default=0.8, type=float, help="control sclae of diffusion model")
    parser.add_argument("--input_image", default="./image_net", type=str, help="input image directory")
    parser.add_argument(
        "--classifier",
        default="resnet50",
        type=str,
        help=f"classification model. Available: {', '.join(IMAGENET_MODEL)}",
    )
    parser.add_argument("--use_hira_adapter", type=str2bool, default=False, help="attach HiRA adapters to the last N transformer blocks of ViT-family backbones")
    parser.add_argument("--hira_expansion_dim", type=int, default=4096, help="hidden expansion dimension for HiRA adapters")
    parser.add_argument("--hira_num_blocks", type=int, default=2, help="number of final transformer MLP blocks that receive HiRA adapters")
    parser.add_argument("--hira_batch_size", type=int, default=32, help="batch size used to fine-tune HiRA weights")
    parser.add_argument("--hira_num_workers", type=int, default=8, help="number of workers used to fine-tune HiRA weights")
    parser.add_argument("--hira_epochs", type=int, default=1, help="number of epochs used to fine-tune HiRA weights")
    parser.add_argument("--hira_lr", type=float, default=1e-4, help="learning rate used to fine-tune HiRA weights")
    parser.add_argument("--hira_weight_decay", type=float, default=1e-4, help="weight decay used to fine-tune HiRA weights")
    parser.add_argument("--hira_seed", type=int, default=0, help="seed used for HiRA fine-tuning and caching")
    parser.add_argument("--hira_cache_dir", type=str, default="pretrained/hira", help="cache directory for fine-tuned HiRA weights")
    parser.add_argument("--hira_dataset_root", type=str, default=None, help="ImageNet root used when fine-tuning HiRA weights")
    parser.add_argument("--hira_max_train_samples", type=int, default=-1, help="optional cap on ImageNet train samples used to fine-tune HiRA")
    parser.add_argument("--hira_force_retrain", type=str2bool, default=False, help="ignore cached HiRA weights and fine-tune again")
    parser.add_argument("--attack_method", default="Linf_pgd", type=str, help="attack model")
    parser.add_argument("--device", default="cuda:0", help="device, e.g. cuda:0")
    parser.add_argument("--use_ranpac_head", type=str2bool, default=False, help="replace the final linear layer with a RanPAC ridge head")
    parser.add_argument("--ranpac_rp_dim", type=int, default=5000, help="random projection dimension for RanPAC")
    parser.add_argument("--ranpac_batch_size", type=int, default=256, help="batch size used to fit the RanPAC head")
    parser.add_argument("--ranpac_num_workers", type=int, default=8, help="number of workers used to fit the RanPAC head")
    parser.add_argument("--ranpac_seed", type=int, default=0, help="seed used to build the RanPAC random projection")
    parser.add_argument(
        "--ranpac_selection_method",
        type=str,
        choices=["regression", "val_acc"],
        default="regression",
        help="which cached RanPAC head to apply after fitting both ridge-selection variants",
    )
    parser.add_argument("--ranpac_cache_dir", type=str, default="pretrained/ranpac", help="cache directory for fitted RanPAC heads")
    parser.add_argument("--ranpac_dataset_root", type=str, default=None, help="ImageNet root used when fitting a RanPAC head")
    parser.add_argument("--stadv_num_iterations", type=int, default=100, help="number of optimization steps for the DiffPure stadv attack")
    parser.add_argument("--stadv_eot_iter", type=int, default=20, help="EOT iterations for the DiffPure stadv attack")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="log final metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="instantpure", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="Weights & Biases entity")
    parser.add_argument("--wandb_name", type=str, default="", help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default="", help="Weights & Biases run group")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Weights & Biases mode: online, offline, or disabled")
    parser.add_argument("--atk_iter", type=int, default=40, help="")
    parser.add_argument("--eps", type=int, default=4, help="")
    args = parser.parse_args()
    return args


def resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def load_diffpure_stadv_attack():
    diffpure_root = Path(__file__).resolve().parent.parent / "DiffPure"
    if not diffpure_root.exists():
        raise ImportError(f"DiffPure was not found at {diffpure_root}.")

    diffpure_root_str = str(diffpure_root)
    if diffpure_root_str not in sys.path:
        sys.path.insert(0, diffpure_root_str)

    from stadv_eot.attacks import StAdvAttack

    return StAdvAttack


def gen_pgd_confs(eps, alpha, iter, input_range=(0, 1)):
    scale = float(input_range[1] - input_range[0]) / 255.0
    return {
        "eps": eps * scale,
        "alpha": alpha * scale,
        "iter": iter,
        "input_range": input_range,
    }


def sample_eval_subset(dataset, num_samples, seed):
    if num_samples >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)

def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict

class Denoised_Classifier(torch.nn.Module):
    def __init__(self, classifier, pipe, device, diffusion, model, t):
        super().__init__()
        self.classifier = classifier
        self.pipe = pipe
        self.device = device
        self.diffusion = diffusion
        self.model = model
        self.t = t
    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t

        x = x * 2 - 1

        t = torch.full((x.shape[0],), t).long().to(x.device)

        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

        indices = list(range(t + 1))[::-1]

        # visualize
        l_sample = []
        l_predxstart = []

        for i in indices:
            # out = self.diffusion.ddim_sample(self.model, sample, t)
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0],), i).long().to(x.device))
   
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def lcm_lora_denoise(self, x, pipe, device, to_512=False):
        batch = x.shape[0]
        prompt = ["" for _ in range(batch)]

        size = x.size()[-1]
        if size != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear")

        start=time.time()

        pil_x = TF.to_pil_image(x.squeeze(0))
        image = np.array(pil_x)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        control_image = control_image.resize((512, 512), resample=Image.NEAREST)

        x = torch.clamp(x, 0, 1)  # assume the input is 0-1
        generator = torch.manual_seed(args.seed)

        image = pipe(
            prompt=prompt,
            image=x,
            control_image=control_image,
            num_inference_steps=args.num_inference_step,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            controlnet_conditioning_scale=args.control_scale,
            generator=generator,
            output_type="pt",
            return_dict=False
        )

        out_image = F.interpolate(image[0], size=(size, size), mode="bilinear")

        end_time = time.time()
        print("run time:", end_time - start)

        return out_image, control_image.resize((size, size))

    def forward(self, x):
        out = self.lcm_lora_denoise(x, self.pipe, self.device)  # [0, 1]
        out = self.classifier(out)
        return out

def generate_x_adv(x, y, classifier, pgd_conf, device):
    net = classifier

    def run_foolbox_pgd(attack_cls, eps, abs_stepsize, criterion):
        fmodel = fb.PyTorchModel(net, bounds=(0, 1), device=device)
        attack = attack_cls(
            steps=pgd_conf["iter"],
            random_start=False,
            abs_stepsize=abs_stepsize,
        )
        _, clipped_advs, _ = attack(fmodel, x, criterion, epsilons=[eps])
        return clipped_advs[0] if isinstance(clipped_advs, (list, tuple)) else clipped_advs[0]

    if args.attack_method == "Linf_pgd":
        x_adv = run_foolbox_pgd(
            fb.attacks.LinfPGD,
            eps=pgd_conf["eps"],
            abs_stepsize=pgd_conf["alpha"],
            criterion=y,
        )

    elif args.attack_method == "L2_pgd":
        x_adv = run_foolbox_pgd(
            fb.attacks.L2PGD,
            eps=0.5,
            abs_stepsize=0.1,
            criterion=y,
        )

    elif args.attack_method == "stadv":
        StAdvAttack = load_diffpure_stadv_attack()
        adversary = StAdvAttack(
            net,
            bound=pgd_conf["eps"],
            num_iterations=args.stadv_num_iterations,
            eot_iter=args.stadv_eot_iter,
        )
        x_adv = adversary(x, y)

    elif args.attack_method == "AutoAttack":
        adversary = AutoAttack(classifier, norm='Linf', eps=pgd_conf["eps"], version='standard', device=device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=1)

    elif args.attack_method == "target_Linf_pgd":
        label_offset = torch.randint(low=1, high=1000, size=y.shape, generator=None).to(device)
        random_target = torch.remainder(y + label_offset, 1000).to(device)
        x_adv = run_foolbox_pgd(
            fb.attacks.LinfPGD,
            eps=pgd_conf["eps"],
            abs_stepsize=pgd_conf["alpha"],
            criterion=fb.criteria.TargetedMisclassification(random_target),
        )

    elif args.attack_method == "snow":
        x_adv = SnowAttack(
            nb_its=10, eps_max=0.0625, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    elif args.attack_method == "fog":
        x_adv = FogAttack(
            nb_its=10, eps_max=128, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    elif args.attack_method == "gabor":
        x_adv = GaborAttack(
            nb_its=10, eps_max=12.5, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    else:
        raise NotImplementedError
    return x_adv.to(device)


def generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t, pipe):
    net = Denoised_Classifier(classifier, pipe, device, diffusion, model, t)

    delta = torch.zeros(x.shape).to(x.device)
    # delta.requires_grad_()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf['eps']
    alpha = pgd_conf['alpha']
    iter = pgd_conf['iter']

    for pgd_iter_id in range(iter):

        x_diff = net.sdedit(x+delta, t).detach()

        x_diff.requires_grad_()

        with torch.enable_grad():
            loss = loss_fn(classifier(x_diff), y)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)
    print("Done")

    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()

def Global(classifier, device, respace, t, args, eps=16, iter=10, name='attack_global', alpha=2, version="v1"):
    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))
    device = resolve_device(device)
    classifier_variant = classifier
    if args.use_hira_adapter:
        classifier_variant = f"{classifier_variant}_hira"
        if args.hira_num_blocks != 2:
            classifier_variant = f"{classifier_variant}_blk{args.hira_num_blocks}"
    if args.use_ranpac_head:
        classifier_variant = f"{classifier_variant}_ranpac_{args.ranpac_selection_method}"
    lora_dir = args.lora_input_dir or "no_lora"

    if args.load_origin_lora:
        save_path = os.path.join(
            args.output_dir,
            f"{args.attack_method}/{classifier_variant}/{args.model}/origin_lora_1/{lora_dir}/num_inference_step_{args.num_inference_step}_strength_{int(args.strength * 1000)}_guidance_scale_{args.guidance_scale}_{args.num_validation_set}_control_scale_{args.control_scale}",
        )
    else:
        save_path = os.path.join(
            args.output_dir,
            f"{args.attack_method}/{classifier_variant}/{args.model}/{lora_dir}/num_inference_step_{args.num_inference_step}_strength_{int(args.strength * 1000)}_guidance_scale_{args.guidance_scale}_{args.num_validation_set}_control_scale_{args.control_scale}",
        )

    os.makedirs(save_path, exist_ok=True)
    wandb_run = init_wandb(args, save_path)
    seed_everything(args.seed)
    classifier = get_archs(
        classifier,
        "imagenet",
        use_hira=args.use_hira_adapter,
        hira_expansion_dim=args.hira_expansion_dim,
        hira_num_blocks=args.hira_num_blocks,
        hira_batch_size=args.hira_batch_size,
        hira_num_workers=args.hira_num_workers,
        hira_epochs=args.hira_epochs,
        hira_lr=args.hira_lr,
        hira_weight_decay=args.hira_weight_decay,
        hira_seed=args.hira_seed,
        hira_cache_dir=args.hira_cache_dir,
        hira_dataset_root=args.hira_dataset_root,
        hira_max_train_samples=args.hira_max_train_samples,
        hira_force_retrain=args.hira_force_retrain,
        use_ranpac=args.use_ranpac_head,
        ranpac_rp_dim=args.ranpac_rp_dim,
        ranpac_batch_size=args.ranpac_batch_size,
        ranpac_num_workers=args.ranpac_num_workers,
        ranpac_seed=args.ranpac_seed,
        ranpac_selection_method=args.ranpac_selection_method,
        ranpac_cache_dir=args.ranpac_cache_dir,
        ranpac_dataset_root=args.ranpac_dataset_root,
        device=device,
    )
    classifier = classifier.to(device)
    classifier.eval()

    dataset = get_dataset(
        'imagenet', split='test', adv=False
    )
    dataset = sample_eval_subset(dataset, args.num_validation_set, args.seed)
    num_eval_samples = len(dataset)

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)

    c = 0

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
    ).to(device, dtype=torch.float32)

    # set scheduler
    if args.model == "LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif args.model == "TCD":
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError("model must be LCM")

    # load LoRA layer and it's weight
    if args.load_origin_lora:
        if args.model == "LCM":
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="origin_lora")
            if args.lora_input_dir != None:
                lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
                unwrapped_state_dict = {}

                for peft_key, weight in lora_state_dict[0].items():
                    key = peft_key.replace("base_model.model.", "")
                    unwrapped_state_dict[key] = weight.to(pipe.dtype)

                pipe.load_lora_weights(unwrapped_state_dict, adapter_name="adv_lora")
                pipe.set_adapters(["origin_lora", "adv_lora"], adapter_weights=[0.5, 0.5])
        elif args.model == "TCD":
            pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="origin_lora")
        else:
            raise ValueError("invalid model")
    else:
        lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
        unwrapped_state_dict = {}

        for peft_key, weight in lora_state_dict[0].items():
            key = peft_key.replace("base_model.model.", "")
            unwrapped_state_dict[key] = weight.to(pipe.dtype)

        pipe.load_lora_weights(unwrapped_state_dict)

    classifier_accuracy = 0
    original_classifier_robust_accuracy = 0
    robust_accuracy = 0
    clean_accuracy = 0

    clean_typical_accuracy = 0
    typical_accuracy = 0

    for subdir in (
        "visualization",
        "clean_image",
        "robust_image",
        "canny_image",
        "adversarial_image",
        "typical_image",
    ):
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)
    
    i = 1

    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)

        classifier_accuracy += (y == classifier(x).argmax(1)).sum().item()

        x_adv_classifier = generate_x_adv(x, y, classifier, pgd_conf, device)
        original_classifier_robust_accuracy += (y == classifier(x_adv_classifier).argmax(1)).sum().item()

        if version == 'v1':
            x_adv = x_adv_classifier
        elif version == 'v2':
            x_adv = generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t, pipe)
        else:
            raise NotImplementedError(f"Unknown attack version: {version}")

        with (torch.no_grad()):
            net = Denoised_Classifier(classifier, pipe, device, diffusion, model, t)

            denoised_clean_x, natual_canny = net.lcm_lora_denoise(x, pipe, device)
            robust_x, adv_canny = net.lcm_lora_denoise(x_adv, pipe, device)
        
        si(x, save_path + f'/clean_image/{i}.png')
        si(robust_x, save_path + f'/robust_image/{i}.png')
        si(x_adv, save_path + f'/adversarial_image/{i}.png')
        
        clean_accuracy += (y == classifier(denoised_clean_x.to(torch.float32)).argmax(1)).sum().item()
        robust_accuracy += (y == classifier(robust_x.to(torch.float32)).argmax(1)).sum().item()
        i += 1

    metrics = {
        "classifier_accuracy": classifier_accuracy / num_eval_samples,
        "original_classifier_robust_accuracy": original_classifier_robust_accuracy / num_eval_samples,
        "attack_fail_rate": original_classifier_robust_accuracy / num_eval_samples,
        "clean_accuracy": clean_accuracy / num_eval_samples,
        "robust_accuracy": robust_accuracy / num_eval_samples,
        "clean_typical_accuracy": clean_typical_accuracy / num_eval_samples,
        "typical_accuracy": typical_accuracy / num_eval_samples,
        "evaluated_examples": i - 1,
    }

    stat = pd.DataFrame(metrics, index=[0])
    stat.to_csv(os.path.join(save_path, "stat.csv"), index=False)

    if wandb_run is not None:
        wandb_run.summary["save_path"] = save_path
        log_wandb_metrics(wandb_run, metrics)
        finish_wandb(wandb_run)

    print(stat)

if __name__ == '__main__':
    args = parse_args()
    Global(args.classifier, args.device, 'ddim50', t=150, eps=args.eps, iter=args.atk_iter, name='attack_global_gradpass', alpha=1,                     #4/255 pgd-100 if want to run autoattack 4/255 just run this and add args.attack_method=AutoAttack
                args=args, version="v1")
