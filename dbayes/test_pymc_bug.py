import pymc
import numpy as np

qH = pymc.Uniform("qH", lower=0.40, upper=0.5, value=0.417)
sigma = pymc.Uniform("sigma", lower=0.31, upper=0.325, value=0.31507524065751241)
epsilon = pymc.Uniform("epsilon", lower=0.5, upper=1.0, value=0.635968)
theta = 1.0

data = [dict(temperature=281.15, density=0.999848)]

def calc(qH, sigma):
    print("qH=%f" % qH)
    if qH < 0:
        raise(ValueError("Wrong sign dude."))    
    return 1.0

temperature = pymc.Uniform("temperatures_%d" % 0, 0.0, 1000.0, value=1.0, observed=True)

density_estimator = pymc.Deterministic(lambda qH, sigma: calc(qH, sigma), "Calculate", "estimator", dict(qH=qH, sigma=sigma))
#@pymc.deterministic(plot=False)
#def d2ensity_estimator(qH=qH, sigma=sigma):
#    print("qH=%f" % qH)
#    if qH < 0:
#        raise(ValueError("Wrong sign dude."))    
#    return 1.0

measurement = pymc.Normal("observed_density_%d" % 0, mu=density_estimator, tau=1.0, value=1.0, observed=True)

variables = [qH, sigma]
variables.append(density_estimator)
variables.append(measurement)

model = pymc.Model(variables)
mcmc = pymc.MCMC(model)

mcmc.sample(10000)

