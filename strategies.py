#!/usr/bin/env python

"""strategies.py contains strategies for Active Learning
Adapted from https://github.com/ej0cl6/deep-active-learning to fit segmentation task in the use case """

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"




import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

class Strategy:
    def __init__(self, dataset, net):
        """
        Initializes the Strategy class.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        self.dataset = dataset
        self.net = net

    def query(self, n):
        """
        Selects a subset of unlabeled samples to query for labeling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        pass

    def update(self, pos_idxs, neg_idxs=None):
        """
        Updates the labeled indices in the dataset.

        Args:
            pos_idxs (ndarray): The indices of positively labeled samples.
            neg_idxs (ndarray or None): The indices of negatively labeled samples. Defaults to None.
        """
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self):
        """
        Trains the network using the labeled data in the dataset.
        """
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        """
        Performs prediction using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The predicted labels.
            ndarray: The ground truth labels.
        """
        preds, masks_gt = self.net.predict(data)
        return preds, masks_gt

    def predict_prob(self, data):
        """
        Calculates the probability predictions using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        """
        Calculates the probability predictions using dropout in the network.

        Args:
            data: The input data.
            n_drop (int): The number of dropout iterations. Defaults to 10.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        """
        Calculates the probability predictions using dropout with split in the network.

        Args:
            data: The input data.
            n_drop (int): The number of dropout iterations. Defaults to 10.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        """
        Calculates the embeddings using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The calculated embeddings.
        """
        embeddings = self.net.get_embeddings(data)
        return embeddings
    

class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        """
        Initializes the RandomSampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n):
        """
        Selects a subset of unlabeled samples randomly for labeling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        return np.random.choice(np.where(self.dataset.labeled_idxs == 0)[0], n, replace=False)


class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        """
        Initializes the EntropySampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on entropy sampling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).view(len(unlabeled_idxs), -1).sum(1)
        top_n_idx = unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
        return top_n_idx
    
class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        """
        Initializes the MarginSampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on margin sampling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs = probs.reshape(len(unlabeled_idxs), -1)
        max_probabilities, _ = probs.max(dim=1)
        min_probabilities, _ = probs.min(dim=1)
        uncertainties = max_probabilities - min_probabilities
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
    
class BALDDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        """
        Initializes the BALDDropout strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
            n_drop (int): The number of dropout iterations. Defaults to 10.
        """
        super(BALDDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on BALD dropout.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        pb = probs.mean(1)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
    
class AdversarialBIM(Strategy):
    def __init__(self, dataset, net, eps=0.05):
        """
        Initializes the AdversarialBIM strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
            eps (float): The epsilon value for the attack. Defaults to 0.05.
        """
        super(AdversarialBIM, self).__init__(dataset, net)
        self.eps = eps

    def cal_dis(self, x):
        """
        Calculates the adversarial distance for a given sample.

        Args:
            x: The input sample.

        Returns:
            float: The adversarial distance.
        """
        nx = torch.unsqueeze(x, 0).cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).cuda()

        out = self.net.clf(nx + eta)
        mask_pred = torch.sigmoid(out)
        ny = mask_pred.round()

        while not torch.equal(mask_pred, ny):
            loss = F.binary_cross_entropy_with_logits(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out = self.net.clf(nx + eta)
            mask_pred = torch.sigmoid(out)
            ny = mask_pred.round()

        return (eta * eta).sum()

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on adversarial BIM.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm.tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.clf.train()

        return unlabeled_idxs[dis.argsort()[:n]]
    
class KCenterGreedy(Strategy):
    def __init__(self, dataset, net):
        """
        Initializes the KCenterGreedy strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(KCenterGreedy, self).__init__(dataset, net)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on K-Center Greedy algorithm.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm.tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

