import os
import logging
import tempfile
import shutil
import argparse
import pandas as pd

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

# TODO(ahmadki):
# batched inference

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', default='xl', type=str)
parser.add_argument('--precision', default='16', type=str)
parser.add_argument('--base-output-dir', default="./output", type=str)
parser.add_argument('--output-dir-name', default=None, type=str)
parser.add_argument('--output-dir-name-postfix', default=None, type=str)
parser.add_argument('--captions-fname', default="captions_5k.tsv", type=str)
parser.add_argument('--guidance', default=8.0, type=float)
parser.add_argument('--scheduler', default="euler", type=str)
parser.add_argument('--steps', default=20, type=int)
parser.add_argument('--negative-prompt', default="normal quality, low quality, worst quality, low res, blurry, nsfw, nude", type=str)
parser.add_argument('--latent-path', default="latents.pt", type=str)
parser.add_argument('--generator-seed', default=None, type=int)
parser.add_argument("--refiner", dest='refiner', action="store_true",
                    help="Whether to add a refiner to the SDXL pipeline."
                          "Applicable only with --model-id=xl")
parser.add_argument("--no-refiner", dest='refiner', action="store_false",
                    help="Whether to add a refiner to the SDXL pipeline."
                          "Applicable only with --model-id=xl")

args = parser.parse_args()

# Init the logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

if args.latent_path and args.generator_seed:
    raise ValueError(
        "Cannot specify both --latent-path and --generator-seed"
    )

if args.model_id == "2":
    args.model_id = "stabilityai/stable-diffusion-2-1"
elif args.model_id == "xl":
    args.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
else:
    raise ValueError(f"{args.model_id} is not a valid model id")

if args.precision == "fp16":
    dtype = torch.float16
elif args.precision == "bf16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32

# Initialize defaults
device = torch.device('cpu')
world_size = 1
rank = 0

# Check for CUDA availability
if torch.cuda.is_available():
    # Check for distributed environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Initialize distributed process group
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        rank = torch.distributed.get_rank()

    device = torch.device('cuda', local_rank)

# load frozen latent
latent_noise = None
if args.latent_path:
    logging.info(f"[{rank}] loading latent from: {args.latent_path}")
    latent_noise = torch.load(args.latent_path).to(dtype)

logging.info(f"[{rank}] args: {args}")
logging.info(f"[{rank}] world_size: {world_size}")
logging.info(f"[{rank}] device: {device}")

logging.info(f"[{rank}] using captions from: {args.captions_fname}")
df = pd.read_csv(args.captions_fname, sep='\t')
logging.info(f"[{rank}] {len(df)} captions loaded")

# split captions among ranks
df = df[rank::world_size]
logging.info(f"[{rank}] {len(df)} captions assigned")

# Build the pipeline
schedulers = {
    "ddpm": DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "ddim": DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "euler_anc": EulerAncestralDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "euler": EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
}
if args.model_id == "stabilityai/stable-diffusion-2-1":
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                   scheduler=schedulers[args.scheduler],
                                                   safety_checker=None,
                                                   add_watermarker=False,
                                                   variant="non_ema",
                                                   torch_dtype=dtype)
else:
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     scheduler=schedulers[args.scheduler],
                                                     safety_checker=None,
                                                     add_watermarker=False,
                                                     variant="fp16" if args.precision == 'fp16' else None,
                                                     torch_dtype=dtype)
    if args.refiner:
        args.model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.model_id,
                                                                        scheduler=schedulers[args.scheduler],
                                                                        safety_checker=None,
                                                                        add_watermarker=False,
                                                                        variant="fp16" if args.precision == 'fp16' else None,
                                                                        torch_dtype=dtype)


pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)
logging.info(f"[{rank}] Pipeline initialized: {pipe}")

if args.refiner:
    refiner_pipe = refiner_pipe.to(device)
    refiner_pipe.set_progress_bar_config(disable=True)
    logging.info(f"[{rank}] Refiner pipeline initialized: {refiner_pipe}")

# Output directory
output_dir = args.output_dir_name or f"{args.model_id.replace('/','--')}__{args.scheduler}__{args.steps}__{args.guidance}__{args.precision}"
if args.output_dir_name_postfix is not None:
    output_dir = f"{output_dir}_{args.output_dir_name_postfix}"

output_dir = os.path.join(args.base_output_dir, output_dir)

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a temporary directory to atomically move the images
tmp_dir = tempfile.mkdtemp()

# Generate the images
for index, row in df.iterrows():
    image_id = row['image_id']
    caption_id = row['id']
    caption_text = row['caption']

    destination_path = os.path.join(output_dir, f"{caption_id}.png")

    # Check if the image already exists in the output directory
    if not os.path.exists(destination_path):
        # Generate the image
        image = pipe(prompt=caption_text,
                     negative_prompt=args.negative_prompt,
                     guidance_scale=args.guidance,
                     generator=torch.Generator(device=device).manual_seed(args.generator_seed) if args.generator_seed else None,
                     latents=latent_noise,
                     num_inference_steps=args.steps).images[0]

        if args.refiner:
            image = refiner_pipe(caption_text,
                                 image=image).images[0]

        # Save the image
        image_path_tmp = os.path.join(tmp_dir, f"{caption_id}.png")
        image.save(image_path_tmp)
        shutil.move(image_path_tmp, destination_path)

        logging.info(f"[{rank}] Saved image {caption_id}: {caption_text}")
