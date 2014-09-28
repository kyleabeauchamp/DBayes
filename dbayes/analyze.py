import pymc

db = pymc.database.hdf5.load("./out2.h5")

s = db.trace("sigma")[:]
e = db.trace("epsilon")[:]

s.mean(), s.std()
e.mean(), e.std()
