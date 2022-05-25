import numpy as np
from chainer import Variable, optimizers
import chainer.functions as F
import felix.src.algorithms.net as net
import six

"""
Most of the code refers to https://github.com/fullflu/learning-to-rank
"""

np.random.seed(71)


# calculate ndcg for k elements (y_true must be non-negative and include at least one non-zero element.)
def ndcg(y_true, y_score, k=100):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    nthres = min(y_true.shape[0], k)
    for i in range(nthres):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.0) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(nthres):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.0) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg


class RankingModel(object):
    def prepare_data_from_candidate_list(self, candidate_dict):
        labels = [
            np.array(
                [
                    candidate.originalQualification
                    for candidate in candidate_dict[query].candidates
                ]
            ).astype(np.float32)
            for query in candidate_dict
        ]
        X = [
            np.array(
                [candidate.features for candidate in candidate_dict[query].candidates]
            ).astype(np.float32)
            for query in candidate_dict
        ]
        return labels, X

    def prepare_train_data_from_candidates_dict(self, candidate_dict):
        self.labels, self.X = self.prepare_data_from_candidate_list(candidate_dict)
        self.num_sessions = len(self.X)
        perm_all = np.random.permutation(self.num_sessions)
        self.train_indices = perm_all[int(self.val_ratio * self.num_sessions) :]
        self.val_indices = perm_all[: int(self.val_ratio * self.num_sessions)]
        self.dim = len(self.X[0][0])

    def prepare_test_data_from_candidates_dict(self, candidate_dict, noscore=True):
        if noscore:
            _, self.test_X = self.prepare_data_from_candidate_list(candidate_dict)
        else:
            self.test_labels, self.test_X = self.prepare_data_from_candidate_list(
                candidate_dict
            )
        self.test_num_sessions = len(self.test_X)


# Listnet class
class ListNet(RankingModel):
    def __init__(
        self,
        n_hidden1=200,
        n_hidden2=78,
        n_hidden3=40,
        batch_size=28,
        max_iter=1000,
        n_thres_cand=40,
        val_ratio=0.5,
        verbose=10,
        early_stopping_waits=10,
    ):
        super(ListNet, self).__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.val_ratio = val_ratio
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_thres_cand = n_thres_cand
        self.early_stopping_waits = early_stopping_waits

    def get_loss(self, x_t, y_t):
        x_t = Variable(x_t)
        y_t = Variable(y_t)
        y_t = F.reshape(y_t, (1, y_t.shape[0]))
        # normalize output score to avoid divergence
        y_t = F.normalize(y_t)
        self.model.cleargrads()
        pred = self.model(x_t)
        # ---- start loss calculation ----
        pred = F.reshape(pred, (pred.shape[1], pred.shape[0]))
        p_true = F.reshape(y_t, (y_t.shape[0], y_t.shape[1]))
        xm = F.max(pred, axis=1, keepdims=True)
        logsumexp = F.logsumexp(pred, axis=1)
        logsumexp = F.broadcast_to(logsumexp, (xm.shape[0], pred.shape[1]))
        loss = -1 * F.sum(p_true * (pred - logsumexp))
        trainres = ndcg(y_t.data, pred.data, self.n_thres_cand)
        if np.isnan(trainres):
            print(y_t.data.max(), y_t.data.min())
        return loss, trainres

    def fit(self, candidate_dict=None):

        self.prepare_train_data_from_candidates_dict(candidate_dict)
        # model initialization

        self.model = net.MLPListNet(
            self.dim, self.n_hidden1, self.n_hidden2, self.n_hidden3
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.alpha = 0.02
        self.optimizer.setup(self.model)
        # start training
        trainres = 0.0
        traincnt = 0
        for iter_ in range(self.max_iter):
            perm_tr = np.random.permutation(self.train_indices)
            for batch_idx in six.moves.range(
                0, self.train_indices.shape[0], self.batch_size
            ):
                loss = 0.0
                for t in perm_tr[batch_idx : batch_idx + self.batch_size]:
                    traincnt += 1
                    sorted_idxes = np.argsort(self.labels[t])[
                        ::-1
                    ]  # get indexes of best items in the beginning of the list
                    nthres = min(self.n_thres_cand, sorted_idxes.shape[0])
                    x_t = self.X[t][sorted_idxes[:nthres]]
                    y_t = self.labels[t][sorted_idxes[:nthres]]
                    loss_t, trainres_t = self.get_loss(x_t, y_t)
                    loss += loss_t
                    trainres += trainres_t
                loss.backward()
                self.optimizer.update()
            if self.verbose:
                if (iter_ + 1) % self.verbose == 0 or iter_ == self.max_iter - 1:
                    print("step:{},train_loss:{}".format(iter_, loss.data))
                    print("train_ndcg:{}".format(trainres / traincnt))
                    trainres = 0.0
                    traincnt = 0
                    if len(self.val_indices) != 0:
                        testres = self.validation()
                        print("valid_ndcg:{}".format(testres / len(self.val_indices)))

    def validation(self):
        testres = 0.0
        for j in self.val_indices:
            sorted_idxes = np.argsort(self.labels[j])[::-1]
            nthres = min(self.n_thres_cand, sorted_idxes.shape[0])
            x_j = Variable(self.X[j][sorted_idxes[:nthres]])
            y_j = Variable(self.labels[j][sorted_idxes[:nthres]])
            y_j = F.reshape(y_j, (1, y_j.shape[0]))
            # normalize output score to avoid divergence
            y_j = F.normalize(y_j)
            pred_j = self.predict(x_j)
            pred_j = F.reshape(pred_j, (pred_j.data.shape[0],))
            testres += ndcg(y_j.data, pred_j.data, self.n_thres_cand)
        return testres

    def predict(self, test_X):
        if test_X.ndim == 2:
            return self.model(test_X)
        else:
            pred = []
            for t, x_t in enumerate(test_X):
                pred_t = self.model(x_t)
                print(pred_t)
                pred.append(pred_t)
            return pred

    def test(self, candidates, noscore=True):

        self.prepare_test_data_from_candidates_dict(
            candidate_dict=candidates, noscore=noscore
        )
        testres = 0
        if noscore:
            for i, query in enumerate(candidates):
                x_i = Variable(self.test_X[i])
                pred_i = self.predict(x_i)
                pred_i = F.reshape(pred_i, (pred_i.data.shape[0],))
                for elem, candidate in zip(pred_i, candidates[query].candidates):
                    candidate.learnedScores = elem.data
                    candidate.qualification = elem.data
            return candidates

        else:
            for i, query in enumerate(candidates):
                x_i = Variable(self.test_X[i])
                y_i = Variable(self.test_labels[i])
                y_i = F.reshape(y_i, (1, y_i.shape[0]))
                # normalize output score to avoid divergence
                y_i = F.normalize(y_i)
                pred_i = self.predict(x_i)
                pred_i = F.reshape(pred_i, (pred_i.data.shape[0],))
                testres += ndcg(y_i.data, pred_i.data, 10)
            print("score:{}".format(testres / self.test_num_sessions))
            return []
