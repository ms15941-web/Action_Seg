import os
import argparse
import numpy as np
import torch
from dataset import get_data_dict, VideoFeatureDataset
from main import Trainer
from utils import load_config_file


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root_data_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='decoder-agg',
                        choices=['encoder', 'decoder-noagg', 'decoder-agg'])
    parser.add_argument('--result_dir', type=str, default='result_eval')
    args = parser.parse_args()

    p = load_config_file(args.config)
    root_data_dir = args.root_data_dir or p['root_data_dir']
    dataset_dir = os.path.join(root_data_dir, p['dataset_name'])

    feature_dir = os.path.join(dataset_dir, 'features')
    label_dir = os.path.join(dataset_dir, 'groundTruth')
    event_list = [i[1] for i in np.loadtxt(os.path.join(dataset_dir, 'mapping.txt'), dtype=str)]

    test_video_list = np.loadtxt(os.path.join(
        dataset_dir, 'splits', f"test.split{p['split_id']}.bundle"), dtype=str)
    test_video_list = [i.split('.')[0] for i in test_video_list]

    test_data_dict = get_data_dict(
        feature_dir=feature_dir,
        label_dir=label_dir,
        video_list=test_video_list,
        event_list=event_list,
        sample_rate=p['sample_rate'],
        temporal_aug=p['temporal_aug'],
        boundary_smooth=p['boundary_smooth']
    )
    test_dataset = VideoFeatureDataset(test_data_dict, len(event_list), mode='test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(dict(p['encoder_params']), dict(p['decoder_params']), dict(p['diffusion_params']),
        event_list, p['sample_rate'], p['temporal_aug'], p['set_sampling_seed'], p['postprocess'],
        device=device
    )
    trainer.model.load_state_dict(torch.load(args.model, map_location=device))

    result_dir = os.path.join(args.result_dir, p['naming'])
    result = trainer.test(test_dataset, args.mode, device, label_dir,
                          result_dir=result_dir, model_path=None)

    for k, v in result.items():
        print(f'{args.mode}-{k}: {v:.2f}')
