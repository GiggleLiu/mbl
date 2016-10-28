'''
View for Two Impurity Kondo model.
'''

from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import eigvalsh
import pdb,time

from tba.hgen import SpinSpaceConfig
from dmrg import DMRGEngine,get_H_bm,get_H
from rglib.hexpand import RGHGen,MaskedEvolutor,NullEvolutor,Evolutor
from rglib.mps import MPS
from nrg.binner import Binner
from blockmatrix import eigbh
from models import *

def get_spec(h,nsite=10):
    '''
    Show the spectrum of heisenberg model.
    '''
    model=HeisenbergModel(J=1.,Jz=1.,h=h,nsite=nsite)
    hgen=SpinHGen(spaceconfig=model.spaceconfig,evolutor=Evolutor(hndim=model.spaceconfig.hndim))
    H,bm=get_H_bm(H=model.H_serial,hgen=hgen,bstr='M')
    #get the Sz=0 block.
    H0=bm.lextract_block(H,0.)
    #Es,bm,H=eigbh(H0,return_evecs=False)
    Es=eigvalsh(H0.toarray())
    return Es

def stat_E(h,nsite=10,nstat=10):
    '''
    show the statistic of energy levels.
    '''
    wlist=linspace(-10,10,1000)
    binner=Binner(bins=linspace(-12,12,10000),tp='linear')
    for i in xrange(nstat):
        print '@%s'%i
        Es=get_spec(h,nsite)
        binner.push(Es,wl=ones(len(Es)))
    spec=binner.get_spec(wlist=wlist,smearing_method='lorenzian',b=20).real
    ion()
    plot(wlist,spec)
    pdb.set_trace()

def stat_dE(h,nsite=10,nstat=10):
    '''
    show the statistic of energy level spacing.
    '''
    NE=2**nsite
    wlist=linspace(0,50./NE,1000)
    binner=Binner(bins=linspace(0.,120./NE,10000),tp='linear')
    for i in xrange(nstat):
        print '@%s'%i
        Es=get_spec(h,nsite)
        binner.push(diff(Es),wl=ones(len(Es)-1))
    spec=binner.get_spec(wlist=wlist,smearing_method='lorenzian',b=10).real
    ion()
    plot(wlist,spec)
    pdb.set_trace()

def stat_dEratio(h,nsite=10,nstat=10):
    '''
    show the statistic of energy level spacing.
    '''
    NE=2**nsite
    wlist=linspace(0,1.,1000)
    binner=Binner(bins=linspace(0.,1.,10000),tp='linear')
    for i in xrange(nstat):
        print '@%s'%i
        dE=diff(get_spec(h,nsite))
        r=dE[1:]/dE[:-1]
        r[r>1.]=1./r[r>1.]
        binner.push(r,wl=ones(len(r)))
        t1=time.time()
    spec=binner.get_spec(wlist=wlist,smearing_method='lorenzian',b=10).real
    ion()
    plot(wlist,spec)
    pdb.set_trace()


def dmrgrun(model):
    '''Get the result from DMRG'''
    model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite)
    #run dmrg to get the initial guess.
    hgen=RGHGen(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='masked')
    dmrgegn=DMRGEngine(hgen=hgen,tol=0,reflect=True)
    dmrgegn.use_U1_symmetry('M',target_block=zeros(1))
    EG,mps=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=30,tol=1e-12)
    return EG,mps

def solve1(h,nsite,config,Jz=1.,J=1.):
    '''
    Run vMPS for Heisenberg model.
    '''
    #generate the model
    if ndim(h)==0:
        hl=cos(618*arange(nsite))*h
    else:
        hl=h
    model=HeisenbergModel(J=J,Jz=Jz,h=hl,nsite=nsite)

    #run vmps
    #generate a random mps as initial vector
    spaceconfig=SpinSpaceConfig([2,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
    H=model.H.use_bm(bmg)
    k0=product_state(config=config,hndim=2,bmg=bmg)

    #setting up the engine
    vegn=VMPSEngine(H=H,k0=k0,eigen_solver='JD')
    vegn.run(endpoint=(6,'->',0),maxN=50,which='SL',nsite_update=2)
    ket=vegn.ket
    filename='data/ket%s_h%sN%s.dat'%(''.join(config.astype(str)),h,nsite)
    ket.save(filename)

def load_mps(h,nsite,config):
    '''Load a MPS'''
    filename='data/ket%s_h%sN%s.dat'%(''.join(config.astype(str)),h,nsite)
    return MPS.load(filename)

def get_projector(h,nsite,c0,c1):
    '''Generate a projection MPO from two kets.'''
    k0=load_mps(h,nsite,c0)
    k1=load_mps(h,nsite,c1)

if __name__=='__main__':
    TestVMPS().test_vmps()


