__author__ = 'sumant'

import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self, likelihood_dist='gaussian'):
        """
        :param str likelihood_dist: The distribution assumed for each Xi (feature) given Y
        :return:
        """

        self.data = {}
        self.params = {}
        self.priors = {}
        self.likelihood_dist = likelihood_dist

    def add_data(self, X, Y):
        """Provide data samples to the classifier for classification.
        :param np.ndarray X: A mxn matrix where m = #samples and n =#features
        :param np.ndarray Y: A mx1 matrix corresponding to the classes
        :return:
        """
        m, n = X.shape
        assert m == Y.shape[0]

        for i in range(m):
            cls = Y[i]
            sample = X[i, :]
            if cls not in self.data:
                self.data[cls] = []
            self.data[cls].append(sample)

        for cls in self.data:
            self.priors[cls] = len(self.data[cls]) / float(m)

        # compute class means and variances
        # Assumptions :
        # 1. Each feature is normally distributed for a given class
        # 2. The standard deviation varies for a given feature depending on the class
        for cls in self.data:
            samples = self.data[cls]
            mean = np.mean(samples, axis=0, keepdims=True)
            stddev = np.std(samples, axis=0, keepdims=True)
            self.params[cls] = (mean, stddev)

    def predict(self, X):
        """
        :param np.ndarray X: A mxn matrix where m = #samples and n =#features
        :return: A list with class labels
        :rtype: np.ndarray
        """
        # TODO: parallelize
        m, n = X.shape
        ret = []
        for i in range(m):
            sample = X[i, :]
            max_prob = -1
            pred_cls = -1
            for cls in self.params:
                mean, stddev = self.params[cls]
                prob = self.priors[cls] * self.get_gaussian_prob(sample, mean, stddev)
                if prob > max_prob:
                    max_prob = prob
                    pred_cls = cls
            ret.append(pred_cls)
        return ret

    @staticmethod
    def get_gaussian_prob(x, mu, sigma):
        exp = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return np.product((1 / (np.sqrt(2 * np.pi) * sigma)) * exp)


if __name__ == '__main__':
    pass
    # nbc = NaiveBayesClassifier()
    # import pandas as pd
    # data = pd.read_csv('data/pima_indians.csv', header=None)
    # train = data.ix[0:500, :]
    # test = data.ix[500:, :]
    # nbc.add_data(train.ix[:, :7].values, train.ix[:, 8].values)
    # pred = nbc.predict(test.ix[:, :7].values)
    # for i in range(len(pred)):
    #     print pred[i], test.iloc[i, 8]
