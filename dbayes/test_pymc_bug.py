import pymc

x = pymc.Uniform("x", lower=0.0, upper=1.0, value=0.5)
y = pymc.Uniform("y", lower=0.0, upper=1.0)
unused = pymc.Uniform("unused", lower=0.0, upper=1.0)

data = [dict(temperature=281.15, density=0.999848)]

def calc(x, y):
    if x < 0:
        print("x=%f" % x)
        raise(ValueError("Wrong sign dude."))    
    return 1.0

temperature = pymc.Uniform("temperatures_%d" % 0, 0.0, 1000.0, value=1.0, observed=True)

density_estimator = pymc.Deterministic(lambda x, y: calc(x, y), "Calculate", "estimator", dict(x=x, y=y))

measurement = pymc.Normal("observed", mu=density_estimator, tau=1.0, value=1.0, observed=True)

variables = [x, y]
variables.append(density_estimator)
variables.append(measurement)

model = pymc.Model(variables)
mcmc = pymc.MCMC(model)

mcmc.sample(50000)

