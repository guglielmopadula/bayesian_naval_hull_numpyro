from jax import vmap
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA
import dill
import jax
from sklearn.metrics import confusion_matrix
import argparse
import os
import time
from sklearn.preprocessing import StandardScaler


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax.lib import xla_bridge
NUM_BAYES_SAMPLES=10#1000
NUM_WARMUP=30#3000
NUM_CHAINS=4
NUM_DATA=600
D_X=5

def invert_permutation(p):
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def run_inference(model, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model,step_size=0.001)
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

'''
# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]
'''
mle_estimates=np.load("./npy_files/naivebayes_coef.npy")
mu_estimates=mle_estimates[:2]
var_estimates=mle_estimates[2:]

'''
def model(X,Y):
    N=X.shape[0]
    D_X=X.shape[-1]
    mu_sigma=var_estimates
    sigma_sigma=0.01*jnp.ones_like(mu_sigma)
    beta_sigma=mu_sigma/sigma_sigma
    alpha_sigma=mu_sigma**2/sigma_sigma
    mu=numpyro.sample("mu",dist.Cauchy(mu_estimates,0.01*jnp.ones((2,D_X))))
    sigma=numpyro.sample("sigma",dist.Gamma(alpha_sigma,beta_sigma))
    x_hat=numpyro.sample("x_hat",dist.Cauchy(mu[Y],sigma[Y]),obs=X) 
    return x_hat   
'''
def predict(mu,sigma,X):
    X=X.reshape(1,NUM_DATA,-1)
    mu0=mu[:,0,:]
    mu1=mu[:,1,:]
    sigma0=sigma[:,0,:]
    sigma1=sigma[:,1,:]
    mu0=mu0.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    mu1=mu1.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    sigma1=sigma1.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    sigma0=sigma0.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    p_true=jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1))/(0.5*jnp.exp(jnp.sum(dist.Normal(mu0,sigma0).log_prob(X),axis=-1))+0.5*jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1)))
    y_pred=dist.Bernoulli(p_true).sample(random.PRNGKey(0))
    return y_pred

def my_log_likehood(posterior_samples,X,Y):
    X=X.reshape(1,NUM_DATA,-1)
    mu=posterior_samples["mu"].reshape(-1,2,D_X)
    sigma=posterior_samples["sigma"].reshape(-1,2,D_X)
    mu0=mu[:,0,:]
    mu1=mu[:,1,:]
    sigma0=sigma[:,0,:]
    sigma1=sigma[:,1,:]
    mu0=mu0.reshape(-1,1,D_X)
    mu1=mu1.reshape(-1,1,D_X)
    sigma1=sigma1.reshape(-1,1,D_X)
    sigma0=sigma0.reshape(-1,1,D_X)
    p_true=np.maximum(np.minimum(jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1))/(0.5*jnp.exp(jnp.sum(dist.Normal(mu0,sigma0).log_prob(X),axis=-1))+0.5*jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1))),0.000001),0.999999)
    return dist.Bernoulli(p_true).log_prob(Y)

def model(X,Y):
    N,D_X=X.shape
    mu_sigma=var_estimates
    sigma_sigma=0.01*jnp.ones_like(mu_sigma)
    beta_sigma=mu_sigma/sigma_sigma
    alpha_sigma=mu_sigma**2/sigma_sigma
    mu=numpyro.sample("mu",dist.Normal(mu_estimates,0.01*jnp.ones((2,D_X))))
    sigma=numpyro.sample("sigma",dist.Gamma(alpha_sigma,beta_sigma))
    x_hat=numpyro.sample("x_hat",dist.Normal(mu[Y],sigma[Y]).to_event(1),obs=X) 
    return x_hat   


def predict(mu,sigma,X):
    X=X.reshape(1,NUM_DATA,-1)
    mu0=mu[:,0,:]
    mu1=mu[:,1,:]
    sigma0=sigma[:,0,:]
    sigma1=sigma[:,1,:]
    mu0=mu0.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    mu1=mu1.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    sigma1=sigma1.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    sigma0=sigma0.reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,1,-1)
    p_true=jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1))/(0.5*jnp.exp(jnp.sum(dist.Normal(mu0,sigma0).log_prob(X),axis=-1))+0.5*jnp.exp(jnp.sum(dist.Normal(mu1,sigma1).log_prob(X),axis=-1)))
    y_pred=dist.Bernoulli(p_true).sample(random.PRNGKey(0))
    return y_pred

# create artificial regression dataset

if __name__ == "__main__":
    print(xla_bridge.get_backend().platform)
    print(jax.device_count())
    print(jax.local_device_count())
    numpyro.set_host_device_count(4)

    data=np.load("./npy_files/data.npy",allow_pickle=True).item()
    target_train=data["target_train"]
    train=data["train"]
    test=data["test"]
    target_test=data["target_test"]





    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    shape=train[0].shape[0]
    posterior_samples,divergences=run_inference(model,rng_key,train,target_train)

    mu=posterior_samples["mu"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,2,-1)
    sigma=posterior_samples["sigma"].reshape(NUM_CHAINS*NUM_BAYES_SAMPLES,2,-1)
    y_fitted_dist=predict(mu,sigma,train)
    y_fitted_prob=np.mean(y_fitted_dist,axis=0)
    y_fitted=np.round(y_fitted_prob)
    y_predictive_dist=predict(mu,sigma,test)
    y_predictive_prob=np.mean(y_predictive_dist,axis=0)
    y_predictive=np.round(y_predictive_prob)
    print(confusion_matrix(target_train,y_fitted))
    print(confusion_matrix(target_test,y_predictive))

    print(my_log_likehood(posterior_samples,train,target_train))

    np.save("./npy_files/bayes_nb_mle.npy",{"posterior_samples":posterior_samples,"y_fitted_dist":y_fitted_dist,"y_predictive_dist":y_predictive_dist,"target_train":target_train,"target_test":target_test,"num_chains":NUM_CHAINS,"num_bayes_samples":NUM_BAYES_SAMPLES,"divergences":divergences})
