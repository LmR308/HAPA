import torch
import torch.nn as nn
import numpy as np
from dqn import DQN
from focal_loss_example import FocalLoss
from model import *
from prepare_data import *
from torch.utils.data import DataLoader
from utils import evaluate
import torch.optim as optim
import os
import random

class RL_model(nn.Module):
    def __init__(self, opt, embedding_dim):
        super(RL_model, self).__init__()
        self.opt = opt
        self.visit_count = 0
        self.singleCapacity = opt.singleCapacity
        _, self.dimension_adj_martix = get_all_reflect_relation(opt.data_path)
        self.aes = AesModel(embedding_dim, opt).cuda()
        
        self.aes_loss_func1 = FocalLoss(opt.FocalLoss_alpha, opt.FocalLoss_gamma,reduction='none')
        self.aes_loss_func2 = nn.L1Loss(reduction='none')
        self.aes_loss_func1_ = FocalLoss(opt.FocalLoss_alpha)
        self.aes_loss_func2_ = nn.L1Loss()
        
        self.aes_cnn_optimizer = optim.Adam(self.aes.CNN.parameters(), lr=opt.lr)
        self.aes_submodels_optimizer = optim.Adam(self.aes.submodels.parameters(), lr=opt.lr)

        self.dqn_state_nums = 7 + opt.embedding_dim
        
        self.dqnlist = [DQN(self.dqn_state_nums, opt.hidden_dim, 2, opt.lr, opt.gamma, opt.epsilon, opt.target_update_nums,
                        opt.ReplayBuffer_capacity, opt.sample_size).cuda() for _ in range(opt.agent_nums)]
        self.train_agents, self.test_agents = {}, {}
        self.ReplayBuffer_capacity = opt.ReplayBuffer_capacity

        self.state_embedding = nn.Embedding(opt.agent_nums, opt.embedding_dim).cuda()
        self.state_embed = self.state_embedding(torch.LongTensor([_ for _ in range(opt.agent_nums)]).cuda()).cuda()
        self.GAT = GAT(self.opt.embedding_dim, self.opt.embedding_dim, self.opt.embedding_dim, 0.7, 0.2, self.opt.num_heads).cuda()

    def cal_s1(self, chosen_dimensions_val):
        std_val, avg_val, min_val, max_val = torch.std(chosen_dimensions_val), torch.mean(chosen_dimensions_val), torch.min(chosen_dimensions_val), torch.max(chosen_dimensions_val)
        m1, m2, m3 = torch.quantile(chosen_dimensions_val, 0.25), torch.quantile(chosen_dimensions_val, 0.50), torch.quantile(chosen_dimensions_val, 0.75)
        return torch.tensor([std_val, avg_val, min_val, max_val, m1, m2, m3], dtype=torch.float).unsqueeze(0).cuda()

    def cal_s2(self, mark_id_list, state_embed):
        s2 = self.state_embed[mark_id_list]
        s2 = torch.mean(s2, dim=0).unsqueeze(0).cuda()
        return s2

    def cal_new_state(self, action_list, aes_ability):
        mark_id_list = []
        for _ in range(len(action_list)):
            if action_list[_] == 1:
                mark_id_list.append(_)
        mark_id_list = torch.LongTensor(mark_id_list)
        s1 = []
        if len(mark_id_list) == 0:
            s1 = torch.tensor([0 for _ in range(7)]).unsqueeze(0).cuda()
        else:
            chosen_dimensions_val = aes_ability[:, mark_id_list]
            s1 = self.cal_s1(chosen_dimensions_val)
        s2 = self.cal_s2(mark_id_list, self.state_embed)
        new_state = torch.cat((s1, s2), dim=1).cuda()
        return new_state

    def init_state(self, aes_ability):
        action_list = [random.randint(0, 1) for _ in range(self.opt.agent_nums)]
        ini_state = self.cal_new_state(action_list, aes_ability)
        return ini_state, action_list

    def cal_new_ability(self, action_list, aes_ability, mark_ability):
        aes_ability, mark_ability, action_list = aes_ability.tolist()[0], mark_ability.tolist()[0], torch.tensor(action_list)
        cur_ability = torch.where(action_list==1, torch.tensor(mark_ability), torch.tensor(aes_ability))

        return torch.tensor(cur_ability).unsqueeze(0).cuda()

    def pearson_correlation(self, martix):
        martix = martix.cpu().detach().numpy()
        correlation_matrix = np.corrcoef(martix)

        upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
        upper_triangle_values = correlation_matrix[upper_triangle_indices]

        average_correlation = np.mean(upper_triangle_values)
        return average_correlation

    def cal_reward(self, last_mse, last_mae, action_list, mark_ability, new_ability, discount):
        mae, mse, rmse, r2, pea, mape, mpe = evaluate(new_ability.tolist(), mark_ability.tolist())
        mse_improve = (last_mse - mse) * self.opt.msepara
        mark_id_list = []
        for _ in range(len(action_list)):
            if action_list[_] == 1:
                mark_id_list.append(_)
        sim = 0
        if len(mark_id_list) < 2:
            sim = 0
        else:
            sim_martix = self.state_embed[torch.tensor(mark_id_list)].cuda()
            sim = self.pearson_correlation(sim_martix)
        punish_val = sum(action_list) * self.opt.mu

        reward = (mse_improve + sim - punish_val) * discount
        return reward

    def RL_train_Aes_Submodels(self, img, labels):
        score = self.aes(img).cuda()
        loss1 = self.aes_loss_func1(score, labels)
        loss2 = self.aes_loss_func2(score, labels)
        loss_vector = (loss1 + loss2)
        return score, loss_vector

    def train_Aes_CNN(self, train_loader):
        self.aes.train()
        for img, labels in train_loader:
            labels = labels.unsqueeze(0).cuda()
            img = img.unsqueeze(0).cuda()
            score = self.aes(img).cuda()
            loss1 = self.aes_loss_func1_(score, torch.tensor(labels).cuda())
            loss2 = self.aes_loss_func2_(score, torch.tensor(labels).cuda())
            loss = loss1 + loss2
            self.aes_cnn_optimizer.zero_grad()
            loss.backward()
            self.aes_cnn_optimizer.step()

    def test(self, test_loader):
        true_list, rec_list = [], []
        with torch.inference_mode():
            for img, name, label in test_loader:
                img = img.cuda()
                score = self.aes(img).cuda()
                rec_list.append(score.tolist())
                true_list.append(label.tolist())
        return evaluate(rec_list, true_list)

    def train_RL(self, train_loader, test_loader):
        num_visits, tl_visits = 0, self.singleCapacity * self.opt.path_len
        self.aes.train()
        self.train_agents = {}
        data_loader = DataLoader(dataset=train_loader, batch_size=self.opt.batchSize, collate_fn=collate_data, drop_last=False)
        while num_visits < tl_visits:
            action_martix, loss_matrix = [], []
            for img, name, labels in train_loader:# img.shape=(3, photo_size, photo_size) labels.shape=(77)
                if name not in self.train_agents:
                    self.train_agents[name] = {}
                    one_picture = img.unsqueeze(0).cuda()# one_picture.shape=(1, 3, photo_size, photo_size)

                    aes_ability = self.aes(one_picture).cuda()# aes_ability.shape=(1, num_classes)

                    self.train_agents[name]['cur_state'], self.train_agents[name]['last_actions'] = self.init_state(aes_ability)
                    self.train_agents[name]['mark_ability'] = torch.tensor(labels).unsqueeze(0).cuda()
                    self.train_agents[name]['new_ability'] = labels

                    self.train_agents[name]['aes_ability'] = aes_ability
                    score, label = aes_ability.tolist(), labels.unsqueeze(0).tolist()
                    mae, mse, rmse, r2, pea, mape, mpe = evaluate(score, label)

                    self.train_agents[name]['last_mse'] = mse
                    self.train_agents[name]['last_mae'] = mae
                    self.train_agents[name]['discount'] = self.opt.gamma
                    self.train_agents[name]['iter_count'] = 0

                if self.train_agents[name]['iter_count'] >= self.opt.path_len:
                    continue
                action_list = [dqn.choose_action(self.train_agents[name]['cur_state']) for dqn in self.dqnlist]
                action_martix.append(torch.tensor(action_list).unsqueeze(0).cuda())

                new_state = self.cal_new_state(action_martix[-1].squeeze(0).tolist(), self.train_agents[name]['aes_ability'])
                self.train_agents[name]['new_state'] = new_state
                self.train_agents[name]['last_actions'] = action_list
                self.train_agents[name]['iter_count'] += 1
                num_visits += 1
            
            action_martix = torch.cat(action_martix, dim=0).cuda()# action_martix.shape=(singleCapacity, num_classes)

            for epoch in range(self.opt.model_epochs):
                new_ability_martix, loss_matrix = [], torch.tensor([], requires_grad=True).cuda()
                for img, name, labels in data_loader:
                    new_ability_vector, loss_vector = self.RL_train_Aes_Submodels(img.cuda(), labels.cuda())
                    loss_matrix = torch.cat((loss_matrix, loss_vector), dim=0)
                    new_ability_martix.append(new_ability_vector)
                
                result_matrix = torch.mm(loss_matrix.T.cuda(), torch.tensor(action_martix, dtype=torch.float32, requires_grad=True).cuda())# result_matrix.shape=(num_classes, num_classes)
                total_loss = result_matrix.sum()
                self.aes_cnn_optimizer.zero_grad()
                total_loss.backward()
                self.aes_cnn_optimizer.step()
                new_ability_martix, loss_matrix = [], torch.tensor([], requires_grad=True).cuda()
                for img, name, labels in data_loader:
                    new_ability_vector, loss_vector = self.RL_train_Aes_Submodels(img.cuda(), labels.cuda())
                    if epoch == self.opt.model_epochs - 1:
                        new_ability_martix.append(new_ability_vector)
                    loss_matrix = torch.cat((loss_matrix, loss_vector), dim=0)
                
                result_matrix = torch.mm(loss_matrix.T.cuda(), torch.tensor(action_martix, dtype=torch.float32, requires_grad=True).cuda())# result_matrix.shape=(num_classes, num_classes)
                total_loss = result_matrix.sum()
                self.aes_submodels_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.aes_submodels_optimizer.step()

                if epoch == self.opt.model_epochs - 1:
                    new_ability_martix = torch.cat(new_ability_martix, dim=0).cuda()
                    for (img, name, labels), action_list, new_ability in zip(train_loader, action_martix, new_ability_martix):
                        reward = self.cal_reward(self.train_agents[name]['last_mse'], self.train_agents[name]['last_mae'], action_list.tolist(), self.train_agents[name]['mark_ability'], new_ability.unsqueeze(0), self.train_agents[name]['discount'])
                        mae, mse, rmse, r2, pea, mape, mpe = evaluate(new_ability.unsqueeze(0).tolist(), self.train_agents[name]['mark_ability'].tolist())
                        self.train_agents[name]['last_mse'], self.train_agents[name]['last_mae'] = mse, mae
                        for idx, dqn in enumerate(self.dqnlist):
                            if action_list[idx].item() == 1 or self.train_agents[name]['last_actions'][idx] == 1:
                                dqn.store_transition(torch.squeeze(self.train_agents[name]['cur_state'], dim=0).tolist(), action_list[idx].item(), reward.item(), torch.squeeze(self.train_agents[name]['new_state'], dim=0).tolist())
                        self.train_agents[name]['last_actions'] = action_list.tolist()
                        for dqn in self.dqnlist:
                            if dqn.memery.size() >= self.opt.min_size:
                                dqn.update()

                        self.train_agents[name]['cur_state'], self.train_agents[name]['discount'] = self.train_agents[name]['new_state'], self.train_agents[name]['discount'] * self.opt.gamma
            mae, mse, rmse, r2, pea, mape, mpe = self.test(test_loader)
            avg_num = torch.sum(action_martix) / self.singleCapacity
            mae, mse, rmse, r2, pea, mape, mpe, avg_num = [metric.item() for metric in (mae, mse, rmse, r2, pea, mape, mpe, avg_num)]

            print(f'{mae} {mse} {rmse} {avg_num}')
            if not os.path.exists(f'./result/'):
                os.mkdir(f"./result/")
            with open(f"./result/{self.opt.expand_name}.txt", mode='a', encoding='utf-8') as file:
                ans = str(f'{mae} {mse} {rmse} {avg_num}\n')
                file.write(ans)

            self.state_embed = self.GAT(self.state_embed, self.dimension_adj_martix).cuda()


    def fit(self, train_loader, test_loader):
        self.train_agents, trainingPictureData, trainedRlData, trainingRlData, data_set, numberOfRlImages = {}, [], [], [], set(), 1000000
        for img, name, labels in train_loader:
            for i in range(len(img)):
                data_set.add(name[i])
                if len(data_set) < self.singleCapacity:
                    trainingPictureData.append([img[i], labels[i]])
                elif len(data_set) == self.singleCapacity:
                    for _ in range(self.opt.model_epochs):
                        self.train_Aes_CNN(trainingPictureData)
                    numberOfRlImages = self.singleCapacity
                elif numberOfRlImages != 0:
                    trainingRlData.append([img[i], name[i], labels[i]])
                    numberOfRlImages -= 1
                else:
                    sampledData = random.sample(trainedRlData, self.singleCapacity - len(trainingRlData)) + trainingRlData
                    self.train_RL(sampledData, test_loader)
                    trainedRlData =  trainedRlData + trainingRlData
                    numberOfRlImages, trainingRlData = int(self.singleCapacity * 0.8), []