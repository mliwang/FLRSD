# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:20:58 2022

@author: 86138
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_
    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings
    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class EmbMarginLoss(nn.Module):
    """ EmbMarginLoss, regularization on embeddings
    """

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def __init__(self):
        # self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'
    
    
class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """
   

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
class TransRec(SequentialRecommender):
    r"""
    TransRec is translation-based model for sequential recommendation.
    It assumes that the `prev. item` + `user`  = `next item`.
    We use the Euclidean Distance to calculate the similarity in this implementation.
    """

    # input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TransRec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']

        # load dataset info
        self.n_users = dataset.user_num

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.bias = nn.Embedding(self.n_items, 1, padding_idx=0)  # Beta popularity bias
        self.T = nn.Parameter(torch.zeros(self.embedding_size))  # average user representation 'global'

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _l2_distance(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1, keepdim=True))  # [B 1]

    def gather_last_items(self, item_seq, gather_index):
        """Gathers the last_item at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1)
        last_items = item_seq.gather(index=gather_index, dim=1)  # [B 1]
        return last_items.squeeze(-1)  # [B]

    def forward(self, user, item_seq, item_seq_len):
        # the last item at the last position
        last_items = self.gather_last_items(item_seq, item_seq_len - 1)  # [B]
        user_emb = self.user_embedding(user)  # [B H]
        last_items_emb = self.item_embedding(last_items)  # [B H]
        T = self.T.expand_as(user_emb)  # [B H]
        seq_output = user_emb + T + last_items_emb  # [B H]
        return seq_output

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        pos_items = interaction[self.POS_ITEM_ID]  # [B]
        neg_items = interaction[self.NEG_ITEM_ID]  # [B] sample 1 negative item

        pos_items_emb = self.item_embedding(pos_items)  # [B H]
        neg_items_emb = self.item_embedding(neg_items)

        pos_bias = self.bias(pos_items)  # [B 1]
        neg_bias = self.bias(neg_items)

        pos_score = pos_bias - self._l2_distance(seq_output, pos_items_emb)
        neg_score = neg_bias - self._l2_distance(seq_output, neg_items_emb)

        bpr_loss = self.bpr_loss(pos_score, neg_score)
        item_emb_loss = self.emb_loss(self.item_embedding(pos_items).detach())
        user_emb_loss = self.emb_loss(self.user_embedding(user).detach())
        bias_emb_loss = self.emb_loss(self.bias(pos_items).detach())

        reg_loss = self.reg_loss(self.T)
        return bpr_loss + item_emb_loss + user_emb_loss + bias_emb_loss + reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)  # [B H]
        test_bias = self.bias(test_item)  # [B 1]

        scores = test_bias - self._l2_distance(seq_output, test_item_emb)  # [B 1]
        scores = scores.squeeze(-1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        test_items_emb = self.item_embedding.weight  # [item_num H]
        test_items_emb = test_items_emb.repeat(seq_output.size(0), 1, 1)  # [user_num item_num H]

        user_hidden = seq_output.unsqueeze(1).expand_as(test_items_emb)  # [user_num item_num H]
        test_bias = self.bias.weight  # [item_num 1]
        test_bias = test_bias.repeat(user_hidden.size(0), 1, 1)  # [user_num item_num 1]

        scores = test_bias - self._l2_distance(user_hidden, test_items_emb)  # [user_num item_num 1]
        scores = scores.squeeze(-1)  # [B n_items]
        return scores