import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

filenames = glob.glob("./symmetric/*.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1).iloc[:, 0]
    if len(x) < 250:
        print(filename)
        continue
    mu = x.values.mean()
    a, b = os.path.splitext(os.path.split(filename)[-1])[0].split("_")
    temperature = float(b)
    chunks = a.split(",")
    parameters = {}
    for chunk in chunks:
        name, parm = chunk.split("=")
        parm = float(parm)
        parameters[name.lstrip()] = parm
    parameters["density"] = mu
    parameters["temperature"] = temperature
    data.append(parameters)
    
data = pd.DataFrame(data)
data.drop("temperature", inplace=True, axis=1)


#X = data[["epsilon0", "epsilon1", "r0", "q0", "sigma0", "sigma1"]].values
X = data[["epsilon0", "q0", "sigma0"]].values
y = data["density"].values

model = sklearn.gaussian_process.GaussianProcess()
model.fit(X, y)
#predict = lambda epsilon0, epsilon1, r0, q0, sigma0, sigma1: np.array(model.predict([epsilon0, epsilon1, r0, q0, sigma0, sigma1], eval_MSE=True))[:, 0]
predict = lambda epsilon0, q0, sigma0: np.array(model.predict([epsilon0, q0, sigma0], eval_MSE=True))[:, 0]

#predict(epsilon0=0.5, epsilon1=0.5, r0=0.1, sigma0=0.2, sigma1=0.2, q0=0.5)
predict(epsilon0=0.5, sigma0=0.2, q0=0.5)


#q0 = pymc.Uniform("q0", 0.0, 0.5)  # Should be symmetric upon q -> - q, so we only need to look at first half of domain.
#sigma0 = pymc.Uniform("sigma0", 0.08, 0.4)
#epsilon0 = pymc.Uniform("epsilon0", 0.2, 2.0)

q0 = pymc.Uniform("q0", 0.1, 0.4)  # Should be symmetric upon q -> - q, so we only need to look at first half of domain.
sigma0 = pymc.Uniform("sigma0", 0.1, 0.3)
epsilon0 = pymc.Uniform("epsilon0", 0.3, 1.5)


#sigma1 = pymc.Uniform("sigma0", 0.08, 0.4)
#epsilon1 = pymc.Uniform("epsilon0", 0.2, 2.0)
sigma1 = 1.0 * sigma0
epsilon1 = 1.0 * epsilon0
r0 = pymc.Uniform("r0", 0.05, 0.25, value=0.2, observed=True)


@pymc.deterministic
def prediction(q0=q0, sigma0=sigma0, epsilon0=epsilon0):
    return predict(q0=q0, sigma0=sigma0, epsilon0=epsilon0)[0]

density_error = 0.001
value = 0.2
experiment = pymc.Normal("observed_density", mu=prediction, tau=density_error ** -2., value=value, observed=True)

variables = [epsilon0, q0, sigma0, prediction, experiment]
mcmc = pymc.MCMC(variables)
mcmc.sample(10000, thin=20, burn=25)
plot(mcmc.trace("q0")[:])
