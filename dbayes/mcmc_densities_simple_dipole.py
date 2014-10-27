import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]

filenames = glob.glob("/home/kyleb/dat/simple_dipoles/*.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1, names=["energy", "density"])
    if len(x) < 250:
        print(filename)
        continue
    mu = x.mean()
    a, b = os.path.splitext(os.path.split(filename)[-1])[0].split("_")
    temperature = float(b)
    chunks = a.split(",")
    parameters = {}
    for chunk in chunks:
        name, parm = chunk.split("=")
        parm = float(parm)
        parameters[name.lstrip()] = parm
    parameters["density"] = mu["density"]
    parameters["energy"] = mu["energy"]
    parameters["temperature"] = temperature
    data.append(parameters)
    
data = pd.DataFrame(data).dropna()
data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["density"])


models = {}
for temperature in data.temperature.unique():
    ind = data.temperature == temperature
    X = data[ind][keys].values
    y = data[ind]["density"].values
    model = sklearn.gaussian_process.GaussianProcess(theta0=1.0)
    model.fit(X, y)
    models[temperature] = model
    #interp = scipy.interpolate.interp2d(X[:, 0], X[:, 1], y, bounds_error=True)
    #models[temperature] = interp

predict = lambda q0, sigma0, temperature: np.array(models[temperature].predict([q0, sigma0], eval_MSE=True))[:, 0]
#predict = lambda q0, sigma0, temperature: np.array(models[temperature](q0, sigma0))
predict(sigma0=0.2, q0=0.5, temperature=320)

q0 = pymc.Uniform("q0", 0.0, 1.0)
sigma0 = pymc.Uniform("sigma0", 0.08, 0.4)


#temperatures = [280, 300, 320]
temperatures = [280, 320]
#temperatures = [300]
@pymc.deterministic
def predictions(q0=q0, sigma0=sigma0):
    try:
        return [predict(q0=q0, sigma0=sigma0, temperature=t)[0] for t in temperatures]
    except ValueError:
        return [100.] * len(temperatures)


#values = np.array([0.038294,  0.038510,  0.038706])
values = np.array([0.038294,  0.038706])
#values = np.array([0.052419])
relative_error = 0.001
density_error = values * relative_error

experiments = pymc.Normal("observed_density", mu=predictions, tau=density_error ** -2., value=values, observed=True)

variables = [q0, sigma0, predictions, experiments]
mcmc = pymc.MCMC(variables)
mcmc.sample(250000, thin=100, burn=5000)

q = mcmc.trace("q0")[:]
s = mcmc.trace("sigma0")[:]

plot(q)
plot(s)
sns.jointplot(q, s, kind="kde")
