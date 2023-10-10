#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
# @Author  ：kinnisoy
# @Date    ：03.28 028 上午 10:13 
# @File    ：generate_pred_parquet.py
# @Description : generate the prediction value in parquet file.
'''
import argparse
import math
import os
import sys
import time

from tqdm import tqdm
import torch
from scipy.special import softmax
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.MFSTMF_CN import MFSTMF
# from model.MMF_v1 import MMF
from typing import Final, List ,Tuple
from init_args import init_args
from av2.utils.typing import NDArrayNumber
# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = root_path+ "/dataset/argoverse2/to_viz/"
# print(save_path)
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser = init_args(parser)

# parser.set_argument("--use_preprocessed", type=bool, default=False)

parser.add_argument("--viz_split", type=str,default=os.path.join(
                root_path, "dataset", "argoverse2", "to_viz"))
parser.add_argument("--ckpt_path", type=str,
                    default="train_log/MFSTMF.ckpt")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PredictionRow = Tuple[str, list]
Prediction: Final[List[str]] = [
    "track_id",
    "pred",
    # "predicted_trajectory_y"
]

def main():
    args = parser.parse_args()

    dataset = ArgoCSVDataset(args.test_split, args.test_split, args)
    data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with weights

    model = MFSTMF.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.eval()

    # Iterate over dataset and generate predictions
    predictions = dict()
    gts = dict()
    cities = dict()
    probabilities = dict()
    final_out = dict()
    times = []
    # for data in tqdm(data_loader):
    for i, data in enumerate(data_loader):
        if i>=100:
            break
        data = dict(data)
        with torch.no_grad():
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            times.append(end_time-start_time)
            output = [x[0:1].detach().cpu().numpy() for x in output]

        # for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
        #     # prediction.shape : (1,6,60,2) prediction.squeeze().shape(6,60,2)
        #     prediction_rows: List[PredictionRow] = []
        #     pred_path = save_path + f"/{argo_id}/" +f"{argo_id}.npz"
        #     prediction_trajs = prediction.squeeze()
        #
        #     p1 = prediction_trajs[0]
        #     p2 = prediction_trajs[1]
        #     p3 = prediction_trajs[2]
        #     p4 = prediction_trajs[3]
        #     p5 = prediction_trajs[4]
        #     p6 = prediction_trajs[5]
        #     np.savez(pred_path,p1,p2,p3,p4,p5,p6)
    print(times)
    np.save('times1.npy',np.array(times))
    print(np.mean(times))

if __name__ == "__main__":
    main()
