# Standard library imports
import csv

# 3rd party imports
import numpy as np
import torch
import os.path
from tqdm import tqdm

# Module imports
from config import get_config
from utils import plot_training_history
from train import train
from inference import inference
from rl_env import StatesGenerator, get_benchmark_rewards


config, _ = get_config()

if not config.inference:
    train(config)

else:
    inference(config)
