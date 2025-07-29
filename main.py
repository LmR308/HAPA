import torch
import random
import numpy as np
import argparse
from model import *
from prepare_data import *
from RL_model import RL_model
import warnings
import time
import os
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.is_available())
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
    parser.add_argument('--hidden_dim', type=int, default=100, help='the hidden state size') 
    parser.add_argument('--model_epochs', type=int, default=20, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--num_layers', type=int, default=6, help='the layers nums')
    parser.add_argument('--num_heads', type=int, default=6, help='the attention heads nums')
    parser.add_argument('--mlp_ratio', type=float, default=1, help='the ratio of hidden layers in the middle')
    parser.add_argument('--Kernel_size1', type=int, default=2, help='the first layer convolution kernel size')
    parser.add_argument('--Kernel_size2', type=int, default=2, help='the second layer convolution kernel size')
    parser.add_argument('--Stride1', type=int, default=2, help='the first layer convolution stride size')
    parser.add_argument('--Stride2', type=int, default=2, help='the second layer convolution stride size')
    parser.add_argument('--num_classes', type=int, default=77, help='the number of categories')
    parser.add_argument('--photo_size', type=int, default=128, help='the size of photo')
    parser.add_argument('--Linear_nums', type=int, default=3, help='the number of Linear')
    parser.add_argument('--data_path', type=str, default='./data', help='the path of data')
    parser.add_argument('--agent_nums', type=int, default=77, help='the number of agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor of DQN')
    parser.add_argument('--epsilon', type=float, default=0.8, help='the probability of greedy exploration by DQN')
    parser.add_argument('--target_update_nums', type=int, default=4, help='the number of iteration cycles for updating the policy network in DQN')
    parser.add_argument('--ReplayBuffer_capacity', type=int, default=80, help='the capacity of ReplayBuffer in DQN')
    parser.add_argument('--min_size', type=int, default=60, help='the minimum data size for experience replay learning in DQN')
    parser.add_argument('--path_len', type=int, default=5, help='the number of steps for exploration and exploitation in reinforcement learning')
    parser.add_argument('--sample_size', type=int, default=20, help='the number of randomly sampled samples in DQN experience replay')
    parser.add_argument('--mu', type=float, default=1, help='the value of mu')
    parser.add_argument('--embedding_dim', type=int, default=200, help='the global vector dimension')
    parser.add_argument('--msepara', type=int, default=1, help='the value of lambda in the reinforcement learning feedback function')
    parser.add_argument('--expand_name', type=str, default=str('HAPA'), help='the extension for storing experimental results')
    parser.add_argument('--singleCapacity', type=int, default=200, help='the size of a single batch of data')
    parser.add_argument('--FocalLoss_alpha', type=float, default=0.75, help='the value of the alpha parameter in FocalLoss')
    parser.add_argument('--FocalLoss_gamma', type=float, default=3, help='the value of the gamma parameter in FocalLoss')
    parser.add_argument('--SEED', type=int, default=1, help='the globally random seed set for the experiment')

    opt = parser.parse_args()
    SEED = opt.SEED
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    train_set, test_set = get_RL_data(opt.photo_size, opt.data_path)
    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True,
                           num_workers=0, drop_last=False)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate_RL, pin_memory=True,
                           num_workers=0, drop_last=False)


    embedding_dim = int((opt.photo_size - opt.Kernel_size1) // opt.Stride1 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int((embedding_dim - opt.Kernel_size2 + 2) // opt.Stride2 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int(embedding_dim * embedding_dim * 3)
    model = RL_model(opt, embedding_dim).cuda()
    start_time = time.time()
    model.fit(train_data, test_data)
    end_time = time.time()
    print(f'cost time is {end_time - start_time}')