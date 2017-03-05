from views import *
ion()

#TIKM_dmrg(J1=1.,J2=0.2,K=0.5,Jp=0.1,nsite=10,which='lanczos')
#stat_E(h=1.,nsite=10,nstat=1000)
#stat_dE(h=1.,nsite=10,nstat=1000)
#stat_dEratio(h=9.,nsite=16,nstat=200)
random.seed(2)
nsite=40
h=8
#c0=random_config(nsite)
#c0=zeros(nsite,dtype='int32')
#c0=array([0,1]*(nsite/2))
#create c1 by fliping two spins
#c1=copy.deepcopy(c0)
#c1[nsite/2]=1-c0[nsite/2]
#c1[nsite/2-1]=1-c0[nsite/2-1]
#c0[2]=c1[2]=1-c0[2]

## Solve one config
#hl=(2*random.random(nsite)-1)*h
#eng1,mps1=solve1(hl=hl,nsite=nsite,config=c0)
#eng2,mps2=solve1(hl=hl,nsite=nsite,config=c1)

## Test Overlap
#E,U=solve_ed(hl=hl,nsite=nsite)
#overlap1,overlap2=abs(mps1.state.conj().dot(U)),abs(mps2.state.conj().dot(U))
#ind1,ind2=argmax(overlap1),argmax(overlap2)
#pdb.set_trace()
#subplot(121)
#show_sz(c0,mps1)
#subplot(122)
#show_sz(c1,mps2)
#ssf(hl=hl,nsite=nsite,config1=c0,config2=c1)
#pdb.set_trace()
#retip(h=h,nsite=nsite)

len_stat_E(h=h,nsite=nsite,nsample=2)
