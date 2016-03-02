__author__ = 'sumant'

class TwoClassLDA(object):
    def __init__(self, class1, class2, class1_label='A', class2_label='B'):
        self.class1_label = class1_label
        self.class2_label = class2_label
        self.class1 = class1
        self.class2 = class2
        self.class1_mean = np.mean(class1, axis=0, keepdims=True)
        self.class2_mean = np.mean(class2, axis=0, keepdims=True)

        self.weights = self._fit()  # column vector of weights

        self.class1_proj = np.dot(self.class1, self.weights)
        self.class2_proj = np.dot(self.class2, self.weights)
        self.c1_mean_proj = np.mean(self.class1_proj)
        self.c2_mean_proj = np.mean(self.class2_proj)
        self.c1_std_proj = np.std(self.class1_proj)
        self.c2_std_proj = np.std(self.class2_proj)

    @staticmethod
    def _get_scatter_matrix(data, mean_vec):
        m, n = data.shape
        zero_mean_mat = data - np.repeat(mean_vec, m, axis=0)
        return np.dot(zero_mean_mat.T, zero_mean_mat)

    def _fit(self):
        """
        :param numpy.ndarray class1: mxn matrix where m = #samples; n = #measurements/sample for
         class 1
        :param numpy.ndarray class2: mxn matrix where m = #samples; n = #measurements/sample for
         class 2
        :param str class1_label: Y-label for class 1
        :param str class2_label: Y-label for class 2
        """
        scatter_class1 = self._get_scatter_matrix(self.class1, self.class1_mean)
        scatter_class2 = self._get_scatter_matrix(self.class2, self.class2_mean)

        within_class_scatter = scatter_class1 + scatter_class2
        mean_diff = self.class1_mean - self.class2_mean
        between_class_scatter = np.dot(mean_diff.T, mean_diff)

        # within_class_scatter += 0.01 * np.eye(within_class_scatter.shape[0])
        # between_class_scatter += 0.01 * np.eye(between_class_scatter.shape[0])

        w, u = np.linalg.eigh(np.dot(np.linalg.inv(within_class_scatter), between_class_scatter))

        # Stored for later access from outisde
        self.eigenvecs = u
        self.eigenvals = w
        return u[:, np.argmax(w)]

    def get_mahalanobis_distance(self):
        return np.sqrt(np.dot((self.class1_mean + self.class2_mean), self.weights))

    def predict(self, points):
        """
        :param numpy.ndarray X: kxn matrix where k=#new samples to classify;
                                n = #measurements/sample
        """
        k, n = points.shape

        # do we need 0 mean pre-condition?
        points_proj = np.dot(self.weights.T, points.T)
        # log_likelihood = np.log(p_c1 / p_c2)
        ret = []
        for i in range(k):
            val = points_proj[0, i]
            r1 = ((val - self.c1_mean_proj) / self.c1_std_proj) ** 2
            r2 = ((val - self.c2_mean_proj) / self.c2_std_proj) ** 2
            prob = np.log(self.class1.shape[0] / self.class2.shape[0]) + \
                np.log((self.c2_std_proj / self.c1_std_proj) * np.exp(-0.5 * (r1 - r2)))
            if prob > 0:
                ret.append(self.class1_label)
            else:
                ret.append(self.class2_label)
        return ret

