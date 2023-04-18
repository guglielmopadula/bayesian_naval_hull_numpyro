import scipy
import numpy as np
np.random.seed(0)
import meshio
from tqdm import trange
from sklearn.decomposition import PCA
import seaborn as sns

alls=np.load("./npy_files/negative_data.npy")
corr=np.corrcoef(alls,rowvar=False)
plot=sns.heatmap(corr,xticklabels=False,yticklabels=False,cmap="coolwarm",vmin=-1,vmax=1,square=True)
plot.set_aspect("equal")
fig = plot.get_figure()
fig.savefig("data_analysis/negative_output.png")
