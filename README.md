# Stable Diffusion Inference

Welcome to the experimental stable diffusion inference repository. This repository aims to test and evaluate the model as a new benchmark candidate for [MLPerf inference](https://mlcommons.org/en/) .

## References
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
- [MLPerf-SSD](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd)
- [MLCommons](https://mlcommons.org/en/)

## Getting Started
### 1. Setting Up the Environment

To set up the environment, build and launch the container using the following commands:

```bash
docker build . -t sd_mlperf_inference
docker run --rm -it --gpus=all -v {PWD}:/workspace sd_mlperf_inference bash
```

**Note** : The subsequent commands are assumed to be executed within this container.

### 2. Dataset Overview
This repository leverages the `coco-2014` validation set for image generation and computation of FID and CLIP scores. [COCO (Common Objects in Context)](https://cocodataset.org/) is a diverse dataset instrumental for object detection, segmentation, and captioning tasks. It boasts a substantial validation set with over 40,000 images encompassing more than 200,000 labels.

For benchmarking purposes, we utilize a random subset of {TBD} images and their associated labels, determined by a preset seed of {TBD}. As the focus is on image generation from labels and subsequent score calculation, downloading the entire COCO dataset is unnecessary. The required files for the benchmark, already part of the repository:
- `captions.tsv`: Contains processed coco2014 validation annotations with 40,504 prompts and their respective IDs. Essential for image generation and CLIP scoring.
- `captions_5k.tsv`: Contains the benchmark annotations. Essential for image generation and CLIP scoring.
- `val2014.npz`: Consists of precomputed statistics of the coco2014 validation set. Utilized for FID scoring.

For details on file generation, refer to **Appendix A** .


### 3. Image Generation
Execute the `main.py` script to generate images:

```bash
python main.py \
    --model-id xl \         # xl for SD-XL, xlr for SD-XL + Refiner
    --guidance 8.0 \
    --precision fp16 \     # fp16, bf16 and fp32
    --scheduler euler \
    --steps 20 \
    --latent-path latents.pt
```

For additional execution options:

```bash
python main.py --help
```


### 4. Compute FID Score

```bash
python fid/fid_score.py \
    --batch-size 1 \                 # batch size for the inception network. keep it 1.
    --subset-size 35000 \            # validation subset size, if you want to score the full dataset don't set the argument
    --shuffle-seed 2023 \            # the seed used for random the random subset selection
    ./val2014.npz \                  # ground truth (coco 2014 validation) statistics
    ./output                         # folder with the generated images
```

For more options:
```bash
python fid/fid_score.py --help
```


### 5. Compute CLIP Score

```bash
python clip/clip_score.py \
    --subset-size 35000 \          # validation subset size, if you want to score the full dataset don't set the argument
    --shuffle-seed 2023 \          # the seed used for random the random subset selection
    --tsv-file captions_5k.tsv \   # captions file
    --image-folder ./output        # Folder with the generated images
    --device cuda                  # Device in which CLIP model is run (cpu, cuda)
```

For more options:

```bash
python clip/clip_score.py --help
```


## Appendix A: Generating Dataset Files

To create the `captions.tsv`, `captions_5k.tsv` and `val2014.npz` files:
1. Download the coco2014 validation set:

```bash
scripts/coco-2014-validation-download.sh
```


2. Process the downloaded annotations (provided in JSON format):

```bash
python process-coco-annotations.py \
    --input-captions-file {PATH_TO_COCO_ANNOTATIONS_FILE} \                 # Input annotations file
    --output-tsv-file captions.tsv \                                        # Output annotations
    --allow-duplicate-images                                                # Pick one prompt per image
```

3. Select a pseduo-random captions subset ():

```bash
python subset_generator.py \
    --seed 2023 \                               # Random number generator seed
    --subset-size 5000 \                        # Subset size
    --input-captions-file captions.tsv \        # Input annotations file
    --output-captions-file captions_5k.tsv      # Output annotations
```


4. Generate ground truth statistics:

```bash

python fid/fid_score.py \
    --batch-size 1 \                      # inception network batch size
    --save-stats {COCO_2014_IMAGES} \     # Input folder with coco images
    val2014.npz                           # Output file
```
