from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_wizard as hw
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set
    train_set = datasets.get_dataset(
        dataset_dict=exp_dict["dataset"],
        split="train",
        datadir=args.datadir,
        exp_dict=exp_dict,
        dataset_size=exp_dict["dataset_size"],
    )
    # test set
    test_set = datasets.get_dataset(
        dataset_dict=exp_dict["dataset"],
        split="test",
        datadir=args.datadir,
        exp_dict=exp_dict,
        dataset_size=exp_dict["dataset_size"],
    )

    test_loader = DataLoader(
        test_set,
        # sampler=val_sampler,
        batch_size=1,
        collate_fn=ut.collate_fn,
        num_workers=args.num_workers,
    )

    # Model
    # ==================
    model = models.get_model(
        model_dict=exp_dict["model"], exp_dict=exp_dict, train_set=train_set
    ).cuda()

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))
    # model.val_score_best = -np.inf

    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    train_sampler = torch.utils.data.RandomSampler(
        train_set,
        replacement=True,
        num_samples=max(len(test_loader.dataset), exp_dict["batch_size"] * 50),
    )

    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        collate_fn=ut.collate_fn,
        batch_size=exp_dict["batch_size"],
        drop_last=True,
        num_workers=args.num_workers,
    )

    for e in range(s_epoch, exp_dict["max_epoch"]):
        # Validate only at the start of each cycle
        score_dict = {}
        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate the model
        val_dict = model.val_on_loader(
            test_loader, savedir_images=os.path.join(savedir, "images"), n_images=3
        )
        score_dict["val_score"] = val_dict["test_score"]

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        # score_df.to_csv(os.path.join(savedir, "score_df.csv"))
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

    print("Experiment completed")


if __name__ == "__main__":
    import exp_configs, job_configs

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--exp_group_list", nargs="+", help="Define which exp groups to run."
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default=None,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-d", "--datadir", default=None, help="Define the dataset directory."
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None, type=str)
    parser.add_argument("-p", "--python_binary_path", default="python")
    parser.add_argument("--num_workers", default=0, type=int)
    args, others = parser.parse_known_args()

    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_configs.JOB_CONFIG,
        job_scheduler=args.job_scheduler,
        python_binary_path=args.python_binary_path,
        use_threads=True,
        args=args,
    )
