import os
import argparse
import sys
import logging

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch import nn
from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.MFSTMF_CN import MFSTMF
from init_args import init_args


# from torchinfo import summary
# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)



def main(args):

    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    val_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        pin_memory=True
    )

    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre, args)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False,
        shuffle=False,  #数据集划分好就不用shuffle ，否则报错
        prefetch_factor = 2
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}-{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=-1,
    )

    model = get_model(args)
    # summary(model,input_size=(3,64), dtypes=['torch.IntTensor'], device='cuda')
    # print(model)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        gpus=args.gpus,
        weights_save_path=None,
        max_epochs=args.num_epochs,
        # precision=32,  # 16-mixed
        accumulate_grad_batches=4,
    )

    trainer.fit(model, train_loader, val_loader)


def test():
    args = parser.parse_args()

    dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args)
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True
    )

def get_model(args):
    if(args.model == 'MFSTMF'):
        model = MFSTMF(args)
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = init_args(parser)
    parser.add_argument("--model", type=str,help='input model name',default='MFSTMF', choices= ['MFSTMF',])
    args = parser.parse_args()
    main(args)
    # test()
