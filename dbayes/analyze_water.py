import pymc

db = pymc.database.hdf5.load("./water.h5")

qH = db.trace("qH")[:]
s = db.trace("sigma")[:]
sH = db.trace("sigmaH")[:]
e = db.trace("epsilon")[:]
r0 = db.trace("r0")[:]
theta = db.trace("theta")[:]


#for key in ["qH", "sigma", "epsilon", "r0", "theta"]:
for key in ["sigma", "sigmaH", "qH"]:
    x = db.trace(key)[:]
    figure()
    plot(x)
    title("%s %f +- %f" % (key, x.mean(), x.std()))
