import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
from sklearn.metrics import pairwise_distances
import dill
import jax
from sklearn.metrics import confusion_matrix
import argparse
import time
from numpyro.infer import log_likelihood

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from jax.lib import xla_bridge
NUM_BAYES_SAMPLES=1000
NUM_WARMUP=3000
NUM_CHAINS=4

def model(X,Y):
    N,D_X=X.shape
    alpha = numpyro.sample("alpha", dist.Normal(0, 5*np.ones((1, 1))))#5
    beta = numpyro.sample("beta", dist.Normal(0,5*jnp.ones((D_X, 1))))#5
    logit=jnp.matmul(X,beta)+alpha
    y=numpyro.sample("Y",dist.BernoulliLogits(logit).to_event(1),obs=Y)
    return y

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.0001) #0.0001
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_BAYES_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=True,
    )
    mcmc.run(rng_key, X, Y)
    divergences=jnp.sum(mcmc.get_extra_fields()["diverging"])
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples(group_by_chain=True),divergences

def my_log_likehood(posterior_samples,train,target_train):
    return log_likelihood(model,posterior_samples,train,target_train)["Y"]


'''
# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]
'''


# create artificial regression dataset




if __name__ == "__main__":
    print(xla_bridge.get_backend().platform)
    print(jax.device_count())
    print(jax.local_device_count())
    numpyro.set_host_device_count(4)

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    data=np.load("./npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]



    classic_coeffs=np.load("./npy_files/classiclogisticregression_coef.npy")
    alpha_mean=classic_coeffs[-1]
    beta_mean=classic_coeffs[:-1]
    alpha_mean=jnp.array(alpha_mean.reshape(1,1))
    beta_mean=jnp.array(beta_mean.reshape(-1,1))


    shape=train[0].shape[0]
    posterior_samples,divergences=run_inference(model,rng_key,train,target_train)
    alpha=posterior_samples["alpha"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)
    beta=posterior_samples["beta"].reshape(NUM_CHAINS,NUM_BAYES_SAMPLES,-1)


    predictive = Predictive(model, posterior_samples, return_sites=["Y"])
    y_fitted_dist=predictive(random.PRNGKey(0), train,None)["Y"]
    y_fitted_dist=y_fitted_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=predictive(random.PRNGKey(0), test,None)["Y"]
    y_predictive_dist=y_predictive_dist.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,-1)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))
    np.save("./npy_files/bayes_logreg_zero.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"num_warmup":NUM_WARMUP,"divergences":divergences})
