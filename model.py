#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2024/8/23 16:16
# @Desc  : p(m,x),p(m|x)p(x)都通过模型学习,增加p(x)的损失函数
import os.path

import torch
import torch.nn as nn

from data_set import DataSet
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class CBPA(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CBPA, self).__init__()
        self.device = args.device
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.embedding_size = args.embedding_size
        self.layers = args.layers
        self.behaviors = args.behaviors
        self.lamb = args.lamb
        self.batch_size = args.batch_size
        self.inter_matrix = dataset.inter_matrix

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embedding.weight.data[1:])
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight.data[1:])

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.cross_loss = nn.BCEWithLogitsLoss()
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_weight = args.reg_weight

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self._construct_graphs()

        self.apply(self._init_weights)

        self._load_model()

    def _construct_graphs(self):
        # 辅助行为图构成
        self.aux_graphs = []
        # 辅助行为和目标行为合并的图
        self.combine_graphs = []
        target_inter_matrix = self.inter_matrix[-1]
        self.target_graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, target_inter_matrix)
        target_inter_matrix_bool = target_inter_matrix.astype(bool)

        for i in range(len(self.behaviors) - 1):
            self.aux_graphs.append(LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, self.inter_matrix[i]))

            tmp_combine_inter_matrix = self.inter_matrix[i].astype(bool)
            tmp_combine_inter_matrix = tmp_combine_inter_matrix + target_inter_matrix_bool
            tmp_combine_inter_matrix = tmp_combine_inter_matrix.astype(float)
            tmp_combine_inter_matrix = tmp_combine_inter_matrix.tocoo()
            self.combine_graphs.append(LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, tmp_combine_inter_matrix))

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def multi_grap_operation(self, embs, graphs):
        all_embeddings = []
        for idx in range(len(graphs)):
            tmp_embeddings = graphs[idx](embs)
            all_embeddings.append(tmp_embeddings)
        return all_embeddings

    def min_max_norm(self, input_tensor):

        min_vals = input_tensor.min(dim=0, keepdim=True).values
        max_vals = input_tensor.max(dim=0, keepdim=True).values
        scaled_tensor = (input_tensor - min_vals) / (max_vals - min_vals + 1e-8)
        sum_vals = scaled_tensor.sum(dim=0, keepdim=True)
        normalized_tensor = scaled_tensor / (sum_vals + 1e-8)
        return normalized_tensor

    def multi_grap_operation(self, embs, graphs):
        all_embeddings = []
        for idx in range(len(graphs)):
            tmp_embeddings = graphs[idx](embs)
            all_embeddings.append(tmp_embeddings)
        return all_embeddings

    def front_door_compute(self, condition_prob, combine_prob, aux_prob):
        sum = condition_prob * torch.sum(combine_prob * aux_prob, dim=0)
        sum = torch.sum(sum, dim=0)
        return sum

    def compute_prob(self, user_embs, item_embs, user, item):
        batch_user_emb = user_embs[user.long()]
        batch_item_emb = item_embs[item.long()]
        score = torch.sum(batch_user_emb * batch_item_emb, dim=-1).squeeze()
        score = torch.relu(score)
        return score

    def forward(self, batch_data):

        user, p_item, n_item = torch.split(batch_data[:, 0], 1, dim=-1)
        aux_user, aux_p_item, aux_n_item = torch.split(batch_data[:, 1:], 1, dim=-1)

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        combine_embs = self.multi_grap_operation(all_embeddings, self.combine_graphs)
        aux_embs = self.multi_grap_operation(all_embeddings, self.aux_graphs)

        combine_p_scores = []
        combine_n_scores = []
        condition_p_scores = []
        condition_n_scores = []
        aux_p_scores = []
        aux_n_scores = []
        aux_loss = []


        for idx in range(len(self.behaviors) - 1):

            tmp_combine_user_all_embedding, tmp_combine_item_all_embedding = torch.split(combine_embs[idx], [self.n_users + 1, self.n_items + 1])
            combine_p_score = self.compute_prob(tmp_combine_user_all_embedding, tmp_combine_item_all_embedding, user, p_item)
            combine_n_score = self.compute_prob(tmp_combine_user_all_embedding, tmp_combine_item_all_embedding, user, n_item)
            combine_p_scores.append(combine_p_score)
            combine_n_scores.append(combine_n_score)

            tmp_aux_user_all_embedding, tmp_aux_item_all_embedding = torch.split(aux_embs[idx], [self.n_users + 1, self.n_items + 1])
            aux_p_score = self.compute_prob(tmp_aux_user_all_embedding, tmp_aux_item_all_embedding, user, p_item)
            aux_n_score = self.compute_prob(tmp_aux_user_all_embedding, tmp_aux_item_all_embedding, user, n_item)
            aux_p_scores.append(aux_p_score)
            aux_n_scores.append(aux_n_score)

            condition_embs = self.target_graph(aux_embs[idx])
            tmp_condition_user_all_embedding, tmp_condition_item_all_embedding = torch.split(condition_embs, [self.n_users + 1, self.n_items + 1])
            condition_p_score = self.compute_prob(tmp_condition_user_all_embedding, tmp_condition_item_all_embedding, user, p_item)
            condition_n_score = self.compute_prob(tmp_condition_user_all_embedding, tmp_condition_item_all_embedding, user, n_item)
            condition_p_scores.append(condition_p_score)
            condition_n_scores.append(condition_n_score)

            aux_user_embs = tmp_aux_user_all_embedding[aux_user[:, idx].long()]
            aux_p_item_emb = tmp_aux_item_all_embedding[aux_p_item[:, idx].long()]
            aux_n_item_emb = tmp_aux_item_all_embedding[aux_n_item[:, idx].long()]
            p_score = torch.sum(aux_user_embs * aux_p_item_emb, dim=-1).squeeze()
            n_score = torch.sum(aux_user_embs * aux_n_item_emb, dim=-1).squeeze()
            tmp_loss = self.bpr_loss(p_score, n_score)
            aux_loss.append(tmp_loss)

        aux_loss = torch.stack(aux_loss)
        aux_loss = torch.mean(aux_loss)

        condition_p_scores = torch.stack(condition_p_scores)
        condition_n_scores = torch.stack(condition_n_scores)
        combine_p_scores = torch.stack(combine_p_scores)
        combine_n_scores = torch.stack(combine_n_scores)
        aux_p_scores = torch.stack(aux_p_scores)
        aux_n_scores = torch.stack(aux_n_scores)
        # aux_p_scores = aux_p_scores / (torch.sum(aux_p_scores, dim=0) + 1e-8)
        # aux_n_scores = aux_n_scores / (torch.sum(aux_n_scores, dim=0) + 1e-8)
        p_scores = self.front_door_compute(condition_p_scores, combine_p_scores, aux_p_scores)
        n_scores = self.front_door_compute(condition_n_scores, combine_n_scores, aux_n_scores)
        rec_loss = self.bpr_loss(p_scores, n_scores)

        total_loss = rec_loss + self.lamb * aux_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            storage_user_embeddings, storage_item_embeddings = [], []
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

            combine_embs = self.multi_grap_operation(all_embeddings, self.combine_graphs)
            aux_embs = self.multi_grap_operation(all_embeddings, self.aux_graphs)

            for idx in range(len(self.behaviors) - 1):
                tmp_combine_user_all_embedding, tmp_combine_item_all_embedding = torch.split(combine_embs[idx], [self.n_users + 1, self.n_items + 1])
                tmp_aux_user_all_embedding, tmp_aux_item_all_embedding = torch.split(aux_embs[idx], [self.n_users + 1, self.n_items + 1])

                condition_embs = self.target_graph(aux_embs[idx])
                tmp_condition_user_all_embedding, tmp_condition_item_all_embedding = torch.split(condition_embs, [self.n_users + 1, self.n_items + 1])

                storage_user_embeddings.append([tmp_condition_user_all_embedding, tmp_combine_user_all_embedding, tmp_aux_user_all_embedding])
                storage_item_embeddings.append([tmp_condition_item_all_embedding, tmp_combine_item_all_embedding, tmp_aux_item_all_embedding])
            self.storage_user_embeddings = storage_user_embeddings
            self.storage_item_embeddings = storage_item_embeddings

        condition_scores = []
        combine_scores = []
        aux_scores = []
        for idx in range(len(self.behaviors) - 1):
            tmp_condition_user_all_embedding, tmp_combine_user_all_embedding, tmp_aux_user_all_embedding = self.storage_user_embeddings[idx]
            tmp_condition_item_all_embedding, tmp_combine_item_all_embedding, tmp_aux_item_all_embedding = self.storage_item_embeddings[idx]
            condition_user_emb = tmp_condition_user_all_embedding[users.long()]
            combine_user_emb = tmp_combine_user_all_embedding[users.long()]
            aux_user_emb = tmp_aux_user_all_embedding[users.long()]
            condition_score = torch.matmul(condition_user_emb, tmp_condition_item_all_embedding.t())
            combine_score = torch.matmul(combine_user_emb, tmp_combine_item_all_embedding.t())
            aux_score = torch.matmul(aux_user_emb, tmp_aux_item_all_embedding.t())
            condition_scores.append(torch.relu(condition_score))
            combine_scores.append(torch.relu(combine_score))
            aux_scores.append(torch.relu(aux_score))

        condition_scores = torch.stack(condition_scores)
        combine_scores = torch.stack(combine_scores)
        aux_scores = torch.stack(aux_scores)
        # aux_scores = aux_scores / (torch.sum(aux_scores, dim=0) + 1e-8)
        rec_score = self.front_door_compute(condition_scores, combine_scores, aux_scores)

        return rec_score