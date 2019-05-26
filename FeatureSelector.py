from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureSelector:

    def __init__(self):
        pass

    def selectAllFeatures(self, features):
        scaler = StandardScaler()  # instantiate
        scaler.fit(features)  # compute the mean and standard which will be used in the next command
        x_scaled = scaler.transform(features)
        # PCA with 150 components
        pca = PCA(n_components=150)
        pca.fit(x_scaled)
        X_pca = pca.transform(x_scaled)

        return X_pca