from sklearn.decomposition import PCA


class FeatureSelection:

    def __init__(self):
        pass

    def selectFeaturesByPCA(self, features):
        # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
        pca = PCA(n_components=10)
        # MLE: maximum likelihood estimation
        # pca = PCA(n_components='mle')
        # fit_transform(X, y=None);
        # X : array-like, shape (n_samples, n_features)
        principalComponents = pca.fit_transform(features)
        return principalComponents