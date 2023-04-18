import numpy as np
true=np.load("npy_files/positive_data.npy")
false=np.load("npy_files/negative_data.npy")
print(true.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(true)
true_red=pca.transform(true)
false_red=pca.transform(false)
print(np.linalg.norm(true-pca.inverse_transform(true_red))/np.linalg.norm(true))
print(np.linalg.norm(false-pca.inverse_transform(false_red))/np.linalg.norm(false))
np.save("npy_files/positive_data_red.npy",true_red)
np.save("npy_files/negative_data_red.npy",false_red)