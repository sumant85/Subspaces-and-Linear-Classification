__author__ = 'sumant'

from sys import maxint

import numpy as np


# TODO: Expand to multi-class LDA
class LDA(object):
    def __init__(self, dataX, labelY):
        self.X = np.copy(dataX)
        self.labels = labelY
        self.data = {}
        """:type: dict{object, list[np.ndarray]}"""
        self.mean_vecs = {}
        self.priors = {}
        self._fit()  # column vector of weights

    def _fit(self):
        """
        :param numpy.ndarray class1: mxn matrix where m = #samples; n = #measurements/sample for
         class 1
        :param numpy.ndarray class2: mxn matrix where m = #samples; n = #measurements/sample for
         class 2
        :param str class1_label: Y-label for class 1
        :param str class2_label: Y-label for class 2
        """
        X = self.X
        m, n = X.shape
        for i in range(m):
            cls = self.labels[i]
            sample = X[i, :]
            if cls not in self.data:
                self.data[cls] = []
            self.data[cls].append(sample)

        net_mean = np.mean(X, axis=0, keepdims=True)
        within_class_scatter = np.zeros((n, n))
        between_class_scatter = np.zeros((n, n))
        for cls in self.data:
            cls_scat = np.zeros((n, n))
            dat = self.data[cls]
            mean = np.mean(dat, axis=0, keepdims=True)
            self.mean_vecs[cls] = mean
            self.priors[cls] = len(dat) / float(X.shape[0])
            for row in dat:
                cls_scat += np.dot((row - mean).T, (row - mean))
            between_class_scatter += len(dat) * np.dot((mean - net_mean).T, (mean - net_mean))
            within_class_scatter += cls_scat

        eigenvals, eigenvecs = np.linalg.eigh(np.linalg.inv(within_class_scatter).
                                              dot(between_class_scatter))

        # Stored for later access from outisde
        desc = np.argsort(eigenvals)[::-1]
        self.eigenvals = eigenvals[desc]
        self.eigenvecs = eigenvecs[:, desc]

        net_mean = np.mean(X, axis=0, keepdims=True)
        X = X - np.repeat(net_mean, X.shape[0], axis=0)
        cov_matrix = np.cov(X.T)
        w, v = np.linalg.eig(cov_matrix)
        self.sphered_data = np.dot(np.dot(np.sqrt(np.linalg.inv(np.diag(w))), v.T), X.T).T
        self.net_mean = net_mean

    def transform(self, X, num_components):
        v = self.eigenvecs
        return np.dot(X, v[:, :min(num_components, len(v))])

    def transform_input(self, num_components):
        return self.transform(self.X, num_components)

    def predict(self, points):
        """
        :param numpy.ndarray X: kxn matrix where k=#new samples to classify;
                                n = #measurements/sample
        """
        points = points - np.repeat(self.net_mean, points.shape[0], axis=0)
        ret = []
        for i in range(points.shape[0]):
            sample = points[i, :]
            pred_class = None
            max_discr_val = -maxint-1
            for cls in self.data:
                mean = self.mean_vecs[cls] - self.net_mean
                # Hasite formula 4.10, substitute I for cov matrix since data is sphered
                discr_val = np.dot(sample, mean.T) - 0.5 * np.dot(mean, mean.T)
                if discr_val > max_discr_val:
                    max_discr_val = discr_val
                    pred_class = cls
            ret.append(pred_class)
        return ret


if __name__ == '__main__':
    class1 = np.matrix('4, 2; 2, 4; 2, 3; 3, 6; 4, 4')
    class2 = np.matrix('9,10; 6,8; 9,5; 8,7; 10,8')
    lda = LDA(np.concatenate((class1, class2), axis=0), np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]))
    print lda.predict(np.matrix('9, 10; 2, 7; 4, 4;, 14, 18'))