import argparse
import os
import sys
from tqdm import tqdm

import torch
from torch.nn import functional as F
from scipy.special import softmax
import numpy as np
import pandas as pd
from av2_validation.challange_submission import ChallengeSubmission

from torch.utils.data import DataLoader
# from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
# from argoverse.evaluation.competition_util import generate_forecasting_h5

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.MMF_v1 import MMF
from model.MMF_v2 import MMF as MMF_v2
from model.MFSTMF_CN import MFSTMF
from model.MFSTMF_wo_ST import MFSTMF_wo_ST
from model.MFSTMF_wo_TS import MFSTMF_wo_TS
from model.TFMF_TGR import TMFModel
from model.crat_pred import CratPred
from model.baseline.LSTM import LSTM_
from init_args import init_args


# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser = init_args(parser)

parser.add_argument("--split", choices=["val", "test"], default="test")
# parser.add_argument("--ckpt_path", type=str, default="train_log/0406-MFSTMF-CN_version_0（FullDataSet）/MFSTMF.ckpt")
parser.add_argument("--model", type=str, default="MFMF_gate",choices = ['MFSTMF', 'CratPred','CratPred-ori','TMFModel','MFMF_bz128',
                                                                      'LSTM_','MFSTMF_wo_ST' ,'MFSTMF_wo_TS','MFMF_v2' ])


ckpt_path = {
    'MFSTMF' : "train_log/0406-MFSTMF-CN_version_0（FullDataSet）/MFSTMF.ckpt",
    'CratPred' : "train_log/0406-CradPred-version_0（FullDataSet）/CradPred.ckpt",
    'CratPred-ori':"checkpoints/bak/Crad_Pred.ckpt",
    'LSTM_':"train_log/0410-LSTM_Residual_version_0/LSTM_Residual.ckpt",
    'MFSTMF_wo_ST' : "train_log/0412-MFSTMF_wo_ST_version_0/MFSTMF_wo_ST.ckpt",
    'MFSTMF_wo_TS' : "train_log/0417-MFSTMF_wo_TS_version_0/MFSTMF_wo_TS.ckpt",
    "MFMF": "train_log/0422-MFMF-EN_version_1/MFMF-EN.ckpt",
    "MFMF_bz128": "train_log/0517-MFMF-EN_version_0/MFMF-EN.ckpt",
    "MFMF_gate": "train_log/0619-MFMF_EN_h64b72_Gate_Fusion_version_1/MFMF_GateFusion.ckpt"
    #  "MFMF_gate": "train_log/0528-MFMF-EN_version_0_FusionST&Gate/MFMF-EN.ckpt"
}


def get_model(args):
    if(args.model == 'MFSTMF'):
        model = MFSTMF.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    # Todo: 新增两个消融模型。
    elif (args.model == 'MFSTMF_wo_ST'):
        model = MFSTMF_wo_ST.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif (args.model == 'MFSTMF_wo_TS'):
        model = MFSTMF_wo_TS.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif(args.model == 'CratPred'):
        model = CratPred.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif(args.model =='TMFModel'):
        model = TMFModel.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif (args.model == 'LSTM_'):
        model = LSTM_.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif(args.model == 'MFMF'):
        model = MMF.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif (args.model == 'MFMF_bz128'):
        model = MMF.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    elif (args.model == 'MFMF_gate'):
        model = MMF_v2.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    else:
        model = CratPred.load_from_checkpoint(checkpoint_path=ckpt_path[args.model])
    print(args.model)
    return model


def main():

    args = parser.parse_args()


    if args.split == "val":
        dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    else:
        dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args)

    data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with weights
    
    # model = MFSTMF.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model = get_model(args)
    model.eval()

    # Iterate over dataset and generate predictions
    predictions = dict()
    gts = dict()
    cities = dict()
    probabilities = dict()
    final_out = dict()
   
    for data in tqdm(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = model(data)
          
            output = [x[0:1].detach().cpu().numpy() for x in output]
        for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
            # prediction.shape : (1,6,60,2) prediction.squeeze().shape(6,60,2)
            predictions[argo_id] = prediction.squeeze()
            sum_1 = np.sum(prediction.squeeze(),axis=1)
            sum_2 = np.sum(sum_1,axis=1)
            sotmax_out = softmax(sum_2) 
            sum_soft = np.sum(sotmax_out)
            if sum_soft > 1 :
                index_max = np.argmax(sotmax_out, axis=0)
                sotmax_out[index_max] = sotmax_out[index_max] - (sum_soft- 1 )
                
            if sum_soft < 1:
                index_min = np.argmin(sotmax_out, axis=0)
                sotmax_out[index_min] = sotmax_out[index_min] + ( 1 - sum_soft )
          
            probabilities[argo_id] = sotmax_out
            cities[argo_id] = data["city"][i]
            gts[argo_id] = data["gt"][i][0] if args.split == "val" else None
          
            # read parquet file and extract track_id for argo_id
            if args.split == "test":
                df = pd.read_parquet(args.test_split + '/' + argo_id + '/scenario_'+argo_id+'.parquet')
                track_id = df['focal_track_id'].values[0]
                track_id_dict = dict()
                track_id_dict[track_id] = [prediction.squeeze(),sotmax_out]
                final_out[argo_id] = track_id_dict
        # break  # when debug for end early

    # Evaluate or submit
    # if args.split == "val":
    #     results_6 = compute_forecasting_metrics(
    #         predictions, gts, cities, 6, 60, 2,probabilities)
    #     results_1 = compute_forecasting_metrics(
    #         predictions, gts, cities, 1, 60, 2,probabilities)
    #     print(results_1)
    #     print(results_6)
    # else:
    if args.split == "test":
        print("Saving...")
        chSubmission = ChallengeSubmission(final_out)
        chSubmission.to_parquet(f'submission/{args.model}.parquet')
        print(f"Saving...At:{root_path} submission/{args.model}.parquet")
   


if __name__ == "__main__":
    main()
