#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
# @Author  ：kinnisoy
# @Date    ：03.27 027 下午 18:38 
# @File    ：init_args.py
# @Description : init_model_args
'''

import os
# Get the paths of the repository
file_path = os.path.abspath(__file__)
# root_path = os.path.dirname(os.path.dirname(file_path)) # server path
root_path = os.path.dirname(file_path) # local path


def init_args(parent_parser):
    parser_dataset = parent_parser.add_argument_group("dataset")
    parser_dataset.add_argument(
        "--train_split", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "train"))
    parser_dataset.add_argument(
        "--val_split", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "val"))
    parser_dataset.add_argument(
        "--test_split", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "test"))
    parser_dataset.add_argument(
        "--train_split_pre", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "train_pre-1.pkl"))
    parser_dataset.add_argument(
        "--val_split_pre", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "val_pre.pkl"))
    parser_dataset.add_argument(
        "--test_split_pre", type=str, default=os.path.join(
            root_path, "dataset", "argoverse2", "test_pre.pkl"))
    parser_dataset.add_argument(
        "--reduce_dataset_size", type=int, default=0)
    parser_dataset.add_argument(
        "--use_preprocessed", type=bool, default=False)
    parser_dataset.add_argument(
        "--align_image_with_target_x", type=bool, default=True)

    parser_training = parent_parser.add_argument_group("training")
    parser_training.add_argument("--num_epochs", type=int, default=36)
    parser_training.add_argument(
        "--lr_values", type=list, default=[1e-3, 1e-4, 1e-3, 1e-4])
    parser_training.add_argument(
        "--lr_step_epochs", type=list, default=[25, 30, 32])
    parser_training.add_argument("--wd", type=float, default=0.01)
    parser_training.add_argument("--batch_size", type=int, default=2)
    parser_training.add_argument("--val_batch_size", type=int, default=2)
    parser_training.add_argument("--workers", type=int, default=0)
    parser_training.add_argument("--val_workers", type=int, default=0)
    parser_training.add_argument("--gpus", type=int, default=1)

    parser_model = parent_parser.add_argument_group("model")
    parser_model.add_argument("--latent_map", type=int, default=64)
    parser_model.add_argument("--latent_size", type=int, default=64)
    parser_model.add_argument("--map2actor_dist", type=int, default=10)  # 6
    parser_model.add_argument("--actor2map_dist", type=int, default=10)  # 7
    parser_model.add_argument("--actor2actor_dist", type=int, default=100)

    parser_model.add_argument("--num_scales", type=int, default=3)  # 多尺度预测 考虑几阶邻居 不同尺度
    parser_model.add_argument("--num_preds", type=int, default=60)
    parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
    parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)

    return parent_parser