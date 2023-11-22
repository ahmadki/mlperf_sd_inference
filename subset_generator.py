import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2023, type=int)
parser.add_argument('--subset-size', default=5000, type=int)
parser.add_argument('--input-captions-file', default="captions.tsv", type=str)
parser.add_argument('--output-captions-file', default="captions_5k.tsv", type=str)
args = parser.parse_args()

# load the TSV file
df = pd.read_csv(args.input_input_captions_file, sep='\t')
# shuffle
df_shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
# filter
df_subset = df_shuffled.iloc[:args.subset_size]
# save
df_subset.to_csv(args.output_captions_file, sep='\t', index=False)
