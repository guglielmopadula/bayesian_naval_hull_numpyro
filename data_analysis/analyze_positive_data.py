import scipy
import numpy as np
np.random.seed(0)
import meshio
from tqdm import trange
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6), 
           dpi = 600) 
 

alls=np.load("./npy_files/positive_data.npy")
corr=np.corrcoef(alls,rowvar=False)
plot=sns.heatmap(corr,xticklabels=False,yticklabels=False,cmap="coolwarm",vmin=-1,vmax=1,square=True)
plot.set_aspect("equal")
fig = plot.get_figure()
fig.savefig("data_analysis/positive_output.png")
