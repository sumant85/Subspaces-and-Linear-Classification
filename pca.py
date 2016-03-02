import numpy as np

from utils import convert_zero_mean


class PCA(object):
    """
    Typical usage:
    pca = PCA(method='svd', whiten=True)
    pca.init(X)
    pca.transform(X)
    pca.transform_input()
    """
    def __init__(self, method='eig', whiten=False):
        """
        :param int num_components: The number of principal components to use. If > n, then
        n components are used.
        :param str method: The method to compute basis vectors
        """
        methods = ['eig', 'svd']
        if method not in methods:
            raise AttributeError('Invalid Method %s', method)
        self.method = method
        self.whiten = whiten
        self.eigenvecs = np.ndarray([])
        self.eigenvals = np.ndarray([])
        self.X = None
        
    def _get_principal_components(self, X):
        m, n = X.shape
        if self.method == 'svd':
            X = X / np.sqrt(m - 1)
            u, w, v = np.linalg.svd(X.T)
        elif self.method == 'eig':
            # compute covariance matrix with unbiased covariance estimation
            cov_mat = np.dot(X.T, X) / (m - 1)
            # compute principal components
            w, u = np.linalg.eig(cov_mat)
        return w, u

    # PCA computation function
    def init(self, X):
        """Runs principal component analysis on the data matrix X and returns
        the eigenvalues sorted in descending order along with corresponding 
        eigenvectors for `num_components` values.
        :param numpy.ndarray X: The data matrix. The columns are the data measurements
        for a given sample and the rows are actual samples. Thus if X is m x n, then 
        m = #samples and n =#measurements per sample
        :rtype: np.ndarray
        :return: Data transformed to the new basis using num_components eigenvectors.
        """
        X = np.copy(X)
        X = convert_zero_mean(X)
        w, v = self._get_principal_components(X)
        desc = np.argsort(w)[::-1]
        self.eigenvecs = v[:, desc]
        self.eigenvals = w[desc]
        self.X = X

    def transform(self, num_components, X):
        w = self.eigenvals
        v = self.eigenvecs
        v = v[:, 0:min(num_components, len(w))]
        # transform data and return
        ret = convert_zero_mean(np.dot(X, v))
        # TODO: Get this to match scikit
        if self.whiten:
            ret = np.dot(ret, np.sqrt(np.diag(np.power(w[:min(num_components, len(w))], -1))))
        return ret

    def transform_input(self, num_components):
        return self.transform(num_components, self.X)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1) # random seed for consistency

    mu_vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

    mu_vec2 = np.array([1,1,1])
    cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T

    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
    pca = PCA(method='svd', whiten=True)
    pca.init(all_samples.T)
    print np.std(pca.transform(2, all_samples.T), axis=0)

    from sklearn.decomposition import PCA as PCA1
    pca = PCA1(n_components=2, whiten=True)
    pca.fit(all_samples.T)
    print np.std(pca.transform(all_samples.T), axis=0)