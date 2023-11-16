import pandas as pd

seed = 2023
subset_size = 5000
input_captions_file = "captions.tsv"
output_captions_file = "captions_5k.tsv"


# load the TSV file
df = pd.read_csv(input_captions_file, sep='\t')
# shuffle
df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
# filter
df_subset = df_shuffled.iloc[:subset_size]
# save
df_subset.to_csv(output_captions_file, sep='\t', index=False)
