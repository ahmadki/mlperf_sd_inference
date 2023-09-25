# Stable Diffusion inference
This is an experimental stable diffusion inference repository that is used to test and evaluate the model as for a new benchmark for MLPerf inference.

[Diffusers](https://huggingface.co/docs/diffusers/index)

[MLPerf-SSD](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd)

[MLCommons](https://mlcommons.org/en/)

## Usage instructions

*This README is a WIP. The build and run scripts and instructions will be improved in the coming weeks*

### Build the container
build and launch the container:
```bash
docker build . -t sd_mlperf_inference
```
You will want to `change target_docker_image` in `./scripts/docker/config.sh`

### Prepare the dataset
(The annotations are provided with the repository, `captions_val2014.json`)
Download the coco-2014 validation annotations.
```bash
./scripts/download_annotations.sh
```

(The processes annotations in tsv format is provided with the repository `captions.tsv`)
The original annotations are provided in json format, include over 200k captions (multiple captions per image) and include additional information that is not relevant to the benchmark.

Run the following script to extract a subset of captions and save them to a tsv file.
```bash
./scripts/extra_annotations.sh
```

If necessary, Download MS-COCO-2014 validation images. This is necessary only if you need to calculate the FID. Even then, the activation weights will be provided in the final reference.
```bash
./scripts/download_dataset.sh
```
