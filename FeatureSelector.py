from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureSelector:

    def __init__(self):
        pass

    def selectFeatures(self, features):

        scaler=StandardScaler()#instantiate
        scaler.fit(features) # compute the mean and standard which will be used in the next command
        x_scaled=scaler.transform(features)# fit and transform can be applied together and I leave that for simple exercise
        # we can check the minimum and maximum of the scaled features which we expect to be 0 and 1


        pca = PCA(n_components=10)
        pca.fit(x_scaled)
        X_pca = pca.transform(x_scaled)

        return X_pca

    def selectAllFeatures(self, features):
        scaler = StandardScaler()  # instantiate
        scaler.fit(features)  # compute the mean and standard which will be used in the next command
        x_scaled = scaler.transform(
            features)  # fit and transform can be applied together and I leave that for simple exercise
        # we can check the minimum and maximum of the scaled features which we expect to be 0 and 1

        pca = PCA(n_components=150)
        pca.fit(x_scaled)
        X_pca = pca.transform(x_scaled)

        return X_pca