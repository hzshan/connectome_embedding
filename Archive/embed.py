import os.path
import sys
import torch
import torch_geometric.data
import torch_geometric.utils
import scipy.sparse

binarize = False
#ftype = "dotprod"
#ftype = "exp"
ftype = "ctype"
#ftype = "mlp_ctype"

D = 5

fname = "data/" + ftype + "_D" + str(D)
if os.path.isfile(fname + ".npz"):
    extension = input("file already exists, type an extension ")
    fname = fname + extension + ".npz"
else:
    fname = fname + ".npz"



if binarize:
    y = torch.tensor(Adj > 0,dtype=torch.float32)
else:
    y = torch.tensor(Adj,dtype=torch.float32)
Jsparse = scipy.sparse.coo_matrix(Adj)

edge_index = torch.LongTensor(np.vstack((Jsparse.row,Jsparse.col)))

edge_attr = torch.tensor(Jsparse.data[:,np.newaxis],dtype=torch.float32) #weights 

x = torch.tensor(types_1hot,dtype=torch.float32)

data = torch_geometric.data.Data(x=x,edge_index=edge_index,edge_attr=edge_attr,num_nodes=N)

z = torch.tensor(np.random.randn(N,D)/np.sqrt(D),dtype=torch.float32,requires_grad=True)

Nepochs = 10000
lr = 0.01
eps_dist = 0.01

if ftype == "dotprod":
    params=[z]
elif ftype == "exp":
    params=[z]
elif ftype == "ctype":
    sig = torch.tensor(np.ones([Ntype,Ntype]),dtype=torch.float32,requires_grad=True)
    A = torch.tensor(0.1*np.ones([Ntype,Ntype]),dtype=torch.float32,requires_grad=True)
    B = torch.tensor(0.5*np.ones([Ntype,Ntype]),dtype=torch.float32,requires_grad=True)
    params=[z,sig,A,B]
elif ftype == "mlp_ctype":
    W1 = torch.tensor(np.random.randn(Ntype,D)/np.sqrt(D),dtype=torch.float32,requires_grad=True)
    W2 = torch.tensor(np.random.randn(D,D)/np.sqrt(D),dtype=torch.float32,requires_grad=True)
    b = torch.zeros(D,requires_grad=True)
    params=[z,W1,W2,B]
else:
    sys.exit("wrong ftype")

optim = torch.optim.Adam(params=params,lr=lr)
Tanh = torch.nn.Tanh()
ReLU = torch.nn.ReLU()
Sigmoid = torch.nn.Sigmoid()
BCELoss = torch.nn.BCELoss()
PoissonNLLLoss = torch.nn.PoissonNLLLoss()

lam = 1. / np.sum([len(x)**2 for x in typeinds])

#calculate loss only on off-diag elements
validinds = np.where((np.ones([N,N]) - np.diag(np.ones(N))).flatten())[0]
yvalid = y.flatten()[validinds]

###
lossa = np.zeros(Nepochs)
for ei in range(Nepochs):
    print(ei)
    optim.zero_grad()

    if ftype == "dotprod":
        w_input = z @ z.T
        reg_loss = 0
    elif ftype == "expdist":
        dsq = torch.sum(z*z,1,keepdim=True) + torch.sum(z*z,1,keepdim=True).T - 2*(z @ z.T)
        w_input =  torch.exp(-dsq)
        reg_loss = 0
    elif ftype == "ctype":
        dsq = torch.sum(z*z,1,keepdim=True) + torch.sum(z*z,1,keepdim=True).T - 2*(z @ z.T)
        x = data.x
        w_input =  (x @ A @ x.T) * torch.exp(-dsq / torch.pow((x @ (sig+eps_dist) @ x.T),2)) + (x @ B @ x.T)
        reg_loss =  - lam*torch.sum(Tanh(dsq)* (x @ x.T))
    elif ftype == "mlp_ctype":
        z2 = ReLU(z + x @ W1 + b) @ W2
        w_input = z2 @ z2.T
        reg_loss = 0

    if binarize:
        loss = BCELoss(Sigmoid(w_input.flatten()[validinds]),yvalid) + reg_loss
    else:
        loss = PoissonNLLLoss(w_input.flatten()[validinds],yvalid) + reg_loss

    loss.backward()
    lossa[ei] = (loss - reg_loss).detach().numpy()
    optim.step()
    torch.clamp(sig,min=0)

w = w_input.detach().numpy()

pred = np.random.poisson(np.exp(  w))
pred = pred - np.diag(np.diag(pred))


if ftype == "ctype":
    np.savez_compressed(fname,w_input=w,lossa=lossa,z=z.detach().numpy(),A=A.detach().numpy(),B=B.detach().numpy(),sig=sig.detach().numpy())
else:
    np.savez_compressed(fname,w_input=w,lossa=lossa,z=z.detach().numpy())
    
