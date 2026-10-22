# FACT Dataset Setup Guide

This guide explains how to prepare new datasets and generate configuration files for training FACT models.

## Dataset Requirements

Your dataset should be organized with the following structure:

```
your_dataset/
├── features/              # Video features (.npy files)
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── groundTruth/          # Frame-level action labels (.txt files) 
│   ├── video1.txt
│   ├── video2.txt
│   └── ...
├── mapping.txt           # Action class mapping file 
└── splits/              # Train/test splits 
    ├── train.split1.bundle
    ├── test.split1.bundle
    └── ...
```

### Required Files

1. **`groundTruth/` folder**: Contains frame-level labels for each video
2. **`mapping.txt`**: Maps action class IDs to action names
3. **`features/` folder**: Pre-computed video features
4. **`splits/` folder**: Predefined train/test splits

## File Formats

### `mapping.txt`
Maps action class indices to human-readable names:
```
0 background
1 take_bowl
2 crack_egg
3 pour_milk
4 stir_mixture
...
```

### `groundTruth/*.txt`
Each file contains frame-level labels, one label per line:
```
background
background
take_bowl
take_bowl
crack_egg
crack_egg
pour_milk
...
```

### `splits/*.bundle`
Lists video names for each split (without file extensions):
```
video1
video3
video5
video7
...
```

### `features/*.npy`
Pre-computed video features stored as NumPy arrays. Each file contains feature vectors for one video:
- **Shape**: `(T, D)` where T=temporal frames, D=feature dimensions
- **Example**: `video1.npy` with shape `(1500, 2048)` for a video with 1500 frames

## Create Configuration File

Once your dataset is properly organized, use the `utils/gen_config.py` script to generate a configuration file. The script helps you set dataset-related parameters, and copy other parameters from a base_config file.

### Example 

```bash
python utils/gen_config.py --dataset_path /path/to/dataset --dataset_name new_dataset --output_config new_dataset.yaml --base_config configs/breakfast.yaml
```

## Some Notes
- `average_transcript_length` is used for computing weight for null class.
  - For one-to-one matching, this variable should be set to the average length of the videos' transcripts in datasets. Transcript means the list of action segments happened in a video (e.g., if a video contains a1->a2->a1->a3, it counts as transcript_length=4).
  - For one-to-many matching, the variable should be set to the average number of unique actions in videos (e.g., if a video contains a1->a2->a1->a3, it counts as 3).
- If videos contain lots of action segments and many of them have repeated action classes, it is recommended to use one-to-many matching. It will largely reduce computation cost.
- The number of token is set as token_ratio * max_transcript_length
  - token_ratio can be between 1.5 and 2. It needs to be >=1.
  - For one-to-one matching, max_transcript_length is set to the longest length of video transcript in the dataset.
  - For one-to-many matching, max_transcript_length is set to the maximal number of unique actions in one video.
- [utils/gen_config.py](utils/gen_config.py) contains code to compute all the mentioned variables.  


