import os
from pathlib import Path
from datetime import datetime
import argparse
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from src.residual_diff import ResidualDiffusion
from src.unets import UnetRes
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

def parse_args():
    p = argparse.ArgumentParser(description="MAPGR training entry.")
    p.add_argument("--dataset", default="dunhuang", choices=["dunhuang", "dhmurals"])
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save_and_sample_every", type=int, default=2000)
    p.add_argument("--sampling_timesteps", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--train_num_steps", type=int, default=100000)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--train_lr", type=float, default=1e-4)
    p.add_argument("--gradient_accumulate_every", type=int, default=4)
    p.add_argument("--input_condition_mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Enable fp16 mixed precision.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable native AMP autocast.")
    p.add_argument("--exp_name", default=os.environ.get("MAPGR_EXP_NAME", ""))
    return p.parse_args()


def get_train_folder(dataset_name: str):
    if dataset_name == "dunhuang":
        return [str(DATA_ROOT / "DUNHUANG" / "train" / "train_GT"), str(DATA_ROOT / "DUNHUANG" / "train" / "train_mask")]
    if dataset_name == "dhmurals":
        return [str(DATA_ROOT / "DhMurals" / "train" / "train_GT"), str(DATA_ROOT / "DhMurals" / "train" / "train_mask")]
    return [str(DATA_ROOT / "muralv2" / "images"), str(DATA_ROOT / "muralv2" / "masks")]


def main():
    args = parse_args()
    exp_name = args.exp_name or f"train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_root = PROJECT_ROOT / "results" / exp_name

    set_seed(args.seed)

    input_condition = True
    input_condition_mask = args.input_condition_mask
    image_size = 256
    num_samples = 1
    sum_scale = 1
    folder = get_train_folder(args.dataset)
    is_dunhuang = args.dataset in {"dunhuang", "dhmurals"}

    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=0,
        input_condition=input_condition
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective='pred_res_noise',
        loss_type='l1',
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask
    )

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=args.train_batch_size,
        num_samples=num_samples,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=0.995,
        amp=args.amp,
        fp16=args.fp16,
        convert_image_to="RGB",
        save_and_sample_every=args.save_and_sample_every,
        results_folder=str(exp_root / "sample"),
        equalizeHist=False,
        crop_patch=True,
        is_dunhuang=is_dunhuang,
    )

    trainer.train()

    if trainer.accelerator.is_local_main_process:
        print(f"[info] dataset: {args.dataset}")
        print(f"[info] experiment outputs: {exp_root}")
        trainer.set_results_folder(str(exp_root / f"test_timestep_{args.sampling_timesteps}"))
        trainer.test(last=False, artifact_root=str(exp_root), artifact_tag=exp_name)
        if is_dunhuang:
            compare_images(
                str(exp_root / f"Ours-DH-{exp_name}"),
                str(exp_root / f"GT-DH-{exp_name}")
            )
        else:
            compare_images(
                str(exp_root / f"Ours-TEN-{exp_name}"),
                str(exp_root / f"GT-TEN-{exp_name}")
            )


if __name__ == "__main__":
    main()
