from random import sample
from sklearn import linear_model

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def OLS(X, Y, W, **options):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model

def Lasso(X, Y, W, lam_backend=0.01, iters_train=1000, **options):
    model = linear_model.Lasso(alpha=lam_backend, fit_intercept=False, max_iter=iters_train)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model

def Ridge(X, Y, W, lam_backend=0.01, iters_train=1000, **options):
    model = linear_model.Ridge(alpha=lam_backend, fit_intercept=False, max_iter=iters_train)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model
