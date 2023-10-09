from multiprocessing import Pool
import multiprocessing
import os
import argparse
import sys
import pickle
import os
import logging
from tqdm import tqdm

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from model.MFSTMF_CN import MFSTMF
from init_args import  init_args
# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser = init_args(parser)

# parser.add_argument("--n_cpus", type=int, default=multiprocessing.cpu_count())  #20
parser.add_argument("--n_cpus", type=int, default=10)
parser.add_argument("--chunksize", type=int, default=16)  # 64位系统必须为16的整数倍  32位 8的整数倍


def preprocess_dataset(dataset, n_cpus, chunksize):
    """Parallely preprocess a dataset to a pickle files
    将数据集并行预处理为pickle文件
    Args:
        dataset: Dataset to be preprocessed
        n_cpus: Number of CPUs to use
        chunksize: Chunksize for parallelization 并行化块大小
    """
    with Pool(n_cpus) as p:
        preprocessed = list(tqdm(p.imap(dataset.__getitem__, [
                            *range(len(dataset))], chunksize), total=len(dataset)))
    # print(len(dataset))
    os.makedirs(os.path.dirname(dataset.input_preprocessed), exist_ok=True)
    with open(dataset.input_preprocessed, 'wb') as f:
        pickle.dump(preprocessed, f)


def main():
    args = parser.parse_args()

    args.use_preprocessed = False

    # train_dataset = ArgoCSVDataset(args.train_split, args.train_split_pre, args)
    # val_dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    test_dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args)
    
    # preprocess_dataset(train_dataset, args.n_cpus, args.chunksize)
    # preprocess_dataset(val_dataset, args.n_cpus, args.chunksize)
    preprocess_dataset(test_dataset, args.n_cpus, args.chunksize)


if __name__ == "__main__":
    main()