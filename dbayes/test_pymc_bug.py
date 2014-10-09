import pymc

x = pymc.Uniform("x", lower=0.0, upper=1.0)
y = pymc.Uniform("y", lower=0.0, upper=1.0)

def calc(x, y):
    if x < 0:
        print("x=%f" % x)
        raise(ValueError("Wrong sign!"))    
    return 1.0

estimator = pymc.Deterministic(eval=lambda x, y: calc(x, y), doc="Calculate", name="estimator", parents=dict(x=x, y=y))
measurement = pymc.Normal("observed", mu=estimator, tau=1.0, value=1.0, observed=True)

variables = [x, y, estimator, measurement]

mcmc = pymc.MCMC(variables)
mcmc.sample(10000)
