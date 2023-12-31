import os
import logging
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from clip_encoder import CLIPEncoder

# Initialize ArgumentParser
parser = argparse.ArgumentParser(description='Calculate CLIP scores for image-caption pairs')
parser.add_argument('--tsv-file', type=str, required=True, help='Path to the TSV file containing image ids and captions')
parser.add_argument('--image-folder', type=str, required=True, help='Path to the folder containing image files')
parser.add_argument('--subset-size', type=int, help='Size of the random subset of data to run the score on', default=None)
parser.add_argument('--shuffle-seed', type=int, help='Seed to pick the random data', default=None)
parser.add_argument('--device', type=str, help='Device to which the model is moved', default='cpu')

# Parse the arguments
args = parser.parse_args()

# Instantiate the CLIPEncoder
clip_encoder = CLIPEncoder(device=args.device)

# Read the TSV file using pandas and store the captions and corresponding image ids
df = pd.read_csv(args.tsv_file, sep='\t')

# If subset_size is provided, select a random subset of the data
if args.subset_size:
    np.random.seed(args.shuffle_seed)
    df = df.sample(n=args.subset_size)

# Calculate the CLIP score for each image-caption pair
clip_scores = []
# Only compute CLIP Scores using generated images (To enable CLIP score computation for partial runs)
id_list = [int(d.split(".")[0]) for d in os.listdir(args.image_folder)]
num_clip_images = 0
for id, caption in zip(df['id'], df['caption']):
    # Check whether sample ID is generated
    if id in id_list:
       # Load the image
       image_path = os.path.join(args.image_folder, f"{id}.png")
       image = Image.open(image_path).convert("RGB")
       clip_score = clip_encoder.get_clip_score(caption, image)

    	# Store the CLIP score
       clip_scores.append(100*clip_score.item())
       num_clip_images+=1

if num_clip_images<len(df):
    logging.warning("{} images missing from output folder".format(len(df)-num_clip_images))


# Print the calculated CLIP scores
print(f"Number of image-caption pairs: {len(clip_scores)}")
final_clip_score = np.mean(clip_scores)
print(f"Final CLIP Score for the {args.image_folder} dataset: {final_clip_score}")
