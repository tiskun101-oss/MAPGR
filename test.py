import argparse
import os
from pathlib import Path

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_diff import ResidualDiffusion
from src.unets import Unet, UnetRes
from src.residual_denoising_diffusion_pytorch import Trainer
from src.utils import set_seed
from metric import compare_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MAPGR_DATA_ROOT",
        PROJECT_ROOT,
    )
)
AGENTS_DIR = Path(__file__).resolve().parent / "agents"


def parse_args():
    parser = argparse.ArgumentParser(description="MAPGR inference entry.")
    parser.add_argument(
        "--dataset",
        default="muralv2",
        choices=["dunhuang", "dhmurals", "muralv2"],
        help="Dataset to run inference on.",
    )
    parser.add_argument("--use_agent_prompt", action="store_true", help="Enable agent-generated prompts for inference only.")
    parser.add_argument(
        "--agent_include_local_damage",
        action="store_true",
        help="Optionally include local (mask-specific) damage hints in agent-generated prompts.",
    )
    parser.add_argument(
        "--agent_out_dir",
        default=str(PROJECT_ROOT / "agent_cache"),
        help="Root directory to save dossier.json and prompt.txt when --use_agent_prompt is set.",
    )
    parser.add_argument(
        "--agent_settings",
        default=str(AGENTS_DIR / "settings.yaml"),
        help="Path to agent settings.yaml.",
    )
    parser.add_argument(
        "--agent_prompts",
        default=str(AGENTS_DIR / "prompts.yaml"),
        help="Path to agent prompts.yaml.",
    )
    parser.add_argument("--last", action="store_true", help="Only save the final sample (default keeps intermediate behavior).")
    parser.add_argument("--ckpt", default="50", help="Checkpoint milestone to load, e.g. 50 -> model-50.pt.")
    return parser.parse_args()


def main():
    args = parse_args()

    # init
    set_seed(10)

    save_and_sample_every = 2000
    # Keep inference config aligned with train.py to avoid state_dict mismatch.
    timesteps = 100
    sampling_timesteps = 20
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 1000

    original_ddim_ddpm = False
    if original_ddim_ddpm:
        condition = False
        input_condition = False
        input_condition_mask = True
    else:
        condition = True
        input_condition = True
        input_condition_mask = True

    if condition:
        if args.dataset == "dunhuang":
            test_gt = DATA_ROOT / "DUNHUANG" / "test" / "test_GT"
            test_mask = DATA_ROOT / "DUNHUANG" / "test" / "test_mask"
            if test_gt.exists() and test_mask.exists():
                folder = [str(test_gt), str(test_mask)]
            else:
                folder = [
                    str(DATA_ROOT / "DUNHUANG" / "train" / "train_GT"),
                    str(DATA_ROOT / "DUNHUANG" / "train" / "train_mask"),
                ]
                print("[warn] DUNHUANG/test not found, fallback to DUNHUANG/train.")
        elif args.dataset == "dhmurals":
            test_gt = DATA_ROOT / "DhMurals" / "test" / "test_GT"
            test_mask = DATA_ROOT / "DhMurals" / "test" / "test_mask"
            if test_gt.exists() and test_mask.exists():
                folder = [str(test_gt), str(test_mask)]
            else:
                folder = [
                    str(DATA_ROOT / "DhMurals" / "train" / "train_GT"),
                    str(DATA_ROOT / "DhMurals" / "train" / "train_mask"),
                ]
                print("[warn] DhMurals/test not found, fallback to DhMurals/train.")
        else:
            folder = [
                str(DATA_ROOT / "muralv2" / "images" / "9-dunhuang-wudai&song"),
                str(DATA_ROOT / "muralv2" / "masks"),
            ]
        train_batch_size = 1
        num_samples = 1
        sum_scale = 1
        image_size = 256
    else:
        folder = str(DATA_ROOT / "dataset" / "CelebA" / "img_align_celeba")
        train_batch_size = 32
        num_samples = 25
        sum_scale = 1
        image_size = 32

    if original_ddim_ddpm:
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
        )
        diffusion = GaussianDiffusion(
            model,
            image_size=image_size,
            timesteps=1000,
            sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
            loss_type="l1",
        )
    else:
        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            share_encoder=0,
            input_condition=input_condition,
        )
        diffusion = ResidualDiffusion(
            model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective="pred_res_noise",
            loss_type="l1",
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
        )

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=1e-4,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        save_and_sample_every=save_and_sample_every,
        equalizeHist=False,
        crop_patch=True,
        is_dunhuang=(args.dataset in {"dunhuang", "dhmurals"}),
        sample_split_mode="all",
        train_split_mode="all",
    )

    # test
    if not trainer.accelerator.is_local_main_process:
        return

    trainer.load(args.ckpt)
    trainer.set_results_folder(f"./results/test_timestep_{sampling_timesteps}_ckpt_{args.ckpt}")
    trainer.test(last=args.last)
    compare_images("./results/Ours-DH/", "./results/GT-DH/")


if __name__ == "__main__":
    main()
