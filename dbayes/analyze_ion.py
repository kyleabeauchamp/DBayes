import pandas as pd

n_molecules = 346

inputs = {
0:"density_300_0.0.csv",
3:"density_300_10.csv",
1:"density_300_02.csv",
2:"density_300_05.csv",
6:"density_300_2.0.csv",
}

data = {}
for n_sodium, filename in inputs.items():
    d = pd.read_csv(filename, names=["energy", "density"], skiprows=1)
    data[n_sodium] = d.mean()


data = pd.DataFrame(data).T
data["n_water"] = n_molecules - 2 * data.index
data["xi"] = data.index / (2. * data.index + data.n_water)
data["relative"] = data.density / data.density[0]

data.plot(style='o', subplots=True)

data[["xi", "relative"]].set_index("xi").plot(style="o")
