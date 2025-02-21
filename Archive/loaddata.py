import pandas as pd
import numpy as np

#load data
datapath = "/home/alk/local/exported-traced-adjacencies-v1.2/"

neuronsall = pd.read_csv(datapath+"traced-neurons.csv")
neuronsall.sort_values(by=['instance'],ignore_index=True,inplace=True)
conns = pd.read_csv(datapath+"traced-total-connections.csv")

Nall = len(neuronsall)
Jall = np.zeros([Nall,Nall],dtype=np.uint)

idhash = dict(zip(neuronsall.bodyId,np.arange(Nall)))
preinds = [idhash[x] for x in conns.bodyId_pre]
postinds = [idhash[x] for x in conns.bodyId_post]

Jall[postinds,preinds] = conns.weight

###
types = np.array(neuronsall.type).astype(str)
#fbt = np.nonzero(["FB" in x for x in types])[0]
#pfn = np.nonzero(["PFN" in x for x in types])[0]
#pfl = np.nonzero(["PFL" in x for x in types])[0]
#hdelta = np.nonzero(["hDelta" in x for x in types])[0]
#vdelta = np.nonzero(["vDelta" in x for x in types])[0]
#othercol = np.nonzero([(("FR" in x) or ("EL" in x) or ("FC" in x) or ("FS" in x))  for x in types])[0]
#col = np.concatenate((hdelta,vdelta,othercol))
def sortsubtype(t,types):
    inds = np.nonzero([t in x for x in types])[0]
    sortinds = np.argsort(types[inds])
    inds = inds[sortinds]
    return inds

types = np.array(neuronsall.type).astype(str)
fbt = sortsubtype("FB",types)
pfn = sortsubtype("PFN",types)
pfl = sortsubtype("PFL",types)
hdelta = sortsubtype("hDelta",types)
vdelta = sortsubtype("vDelta",types)
othercol = np.concatenate((sortsubtype("FR",types),sortsubtype("EL",types),sortsubtype("FC",types),sortsubtype("FS",types)))


allcx = np.concatenate((col,fbt,pfn,pfl))

###
J = Jall[allcx,:][:,allcx]
N = J.shape[0]
neurons = neuronsall.iloc[allcx,:]
neurons.reset_index(inplace=True)

uniqtypes = pd.unique(neurons.type)
Ntype = len(uniqtypes)
typehash = dict(zip(uniqtypes,np.arange(Ntype)))
typeclasses = np.array([typehash[x] for x in neurons.type])
typeinds = [np.nonzero(neurons.type == uniqtypes[ii])[0] for ii in range(Ntype)]

types_1hot = np.zeros([N,Ntype])
types_1hot[np.arange(N),typeclasses] = 1.

Adj = J.T #adjacency matrix in row presynaptic, column postsynaptic format
