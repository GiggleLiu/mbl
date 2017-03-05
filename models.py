'''
Heisenberg Model.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
import pdb,time,copy

from tba.hgen import SpinSpaceConfig
from pymps import MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,product_state,random_product_state
from pymps.ssf import SSFLR
from dmrg import DMRGEngine,VMPSEngine
from blockmatrix import SimpleBMG

class HeisenbergModel():
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz:
        Exchange interaction at xy direction and z direction.
    h:
        The strength of weiss field.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J,Jz,h,nsite):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=OpUnitI(hndim=2)
        Sx=opunit_Sx(spaceconfig=self.spaceconfig)
        Sy=opunit_Sy(spaceconfig=self.spaceconfig)
        Sz=opunit_Sz(spaceconfig=self.spaceconfig)

        SL,SZ=[],[]
        for i in xrange(nsite):
            Sxi,Syi,Szi=copy.copy(Sx),copy.copy(Sy),copy.copy(Sz)
            Sxi.siteindex=i
            Syi.siteindex=i
            Szi.siteindex=i
            SL.append(array([Sxi,Syi,sqrt(Jz/J)*Szi]))
            SZ.append(Szi)
        ops=[]
        for i in xrange(nsite):
            if i!=nsite-1:
                ops.append(J*SL[i].dot(SL[i+1]))
            ops.append(h[i]*SZ[i])

        mpc=sum(ops)
        self.H_serial=mpc
        mpo=mpc.toMPO(method='direct')
        mpo.compress()
        self.H=mpo

class HeisenbergModel2(object):
    '''
    Heisenberg model application for vMPS with second nearest neighbor interaction.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz/J2/Jz2:
        Exchange interaction at xy direction and z direction, and the same parameter for second nearest hopping term.
    h:
        The strength of weiss field.
    '''
    def __init__(self,J,Jz,h,nsite,J2=0,J2z=None):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=OpUnitI(hndim=2)
        Sx=opunit_Sx(spaceconfig=self.spaceconfig)
        Sy=opunit_Sy(spaceconfig=self.spaceconfig)
        Sz=opunit_Sz(spaceconfig=self.spaceconfig)

        #with second nearest hopping terms.
        if J2==0:
            self.H_serial2=None
            return
        elif J2z==None:
            J2z=J2
        SL=[]
        for i in xrange(nsite):
            Sxi,Syi,Szi=copy.copy(Sx),copy.copy(Sy),copy.copy(Sz)
            Sxi.siteindex=i
            Syi.siteindex=i
            Szi.siteindex=i
            SL.append(array([Sxi,Syi,sqrt(J2z/J2)*Szi]))
        ops=[]
        for i in xrange(nsite-1):
            ops.append(J*SL[i].dot(SL[i+1]))
            if i<nsite-2 and J2z!=0:
                ops.append(J2*SL[i].dot(SL[i+2]))

        mpc=sum(ops)
        self.H_serial=mpc
        ion()
        mpc.show_advanced()
        pdb.set_trace()

class TIKM(object):
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian: see 10.1038/ncomms4784

    Construct
    -----------------
    HeisenbergModel(J1,J2,K,Jp)

    Attributes
    ------------------
    J1/J2:
        Exchange interaction of nearest hopping and next nearest hopping.
    K/Jp:
        The modulation factor for inter-impurity interaction and impurity-bath interaction.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J1,J2,K,Jp,nsite,impurity_site=None):
        self.spaceconfig=SpinSpaceConfig([2,1])
        if impurity_site is None:
            impurity_site=nsite/2  #nsite/2 and nsite/2-1
        self.impurity_site=impurity_site
        self.J1,self.J2,self.K,self.Jp=J1,J2,K,Jp
        I=OpUnitI(hndim=self.spaceconfig.hndim)
        Sx=opunit_Sx(spaceconfig=self.spaceconfig)
        Sy=opunit_Sy(spaceconfig=self.spaceconfig)
        Sz=opunit_Sz(spaceconfig=self.spaceconfig)
        SL=[]
        for i in xrange(nsite):
            Sxi,Syi,Szi=copy.deepcopy(Sx),copy.deepcopy(Sy),copy.deepcopy(Sz)
            Sxi.siteindex=i
            Syi.siteindex=i
            Szi.siteindex=i
            SL.append(array([Sxi,Syi,Szi]))
        ops=[]
        for i in xrange(nsite-1):
            factor1=J1
            if i==impurity_site or i==impurity_site-2:
                factor1*=Jp
            elif i==impurity_site-1:
                factor1*=K
            ops.append(factor1*SL[i].dot(SL[i+1]))
            if i<nsite-2 and i!=impurity_site-1 and i!=impurity_site-2:
                factor2=J2
                if i==impurity_site-3 or i==impurity_site:
                    factor2*=Jp
                ops.append(factor2*SL[i].dot(SL[i+2]))

        mpc=sum(ops)
        self.H_serial=mpc

def solve1(hl,nsite,config,Jz=1.,J=1.,save=True):
    '''
    Run vMPS for Heisenberg model.
    '''
    #generate the model
    print 'SOLVING %s'%config
    model=HeisenbergModel(J=J,Jz=Jz,h=hl,nsite=nsite)

    #run vmps
    #generate a random mps as initial vector
    spaceconfig=SpinSpaceConfig([2,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
    H=model.H.use_bm(bmg)
    k0=product_state(config=config,hndim=2,bmg=bmg)

    #setting up the engine
    vegn=VMPSEngine(H=H,k0=k0.toket(),eigen_solver='JD')
    eng,ket=vegn.run(endpoint=(6,'->',0),maxN=20,which='SL',nsite_update=2)
    #save to file
    if save:
        ind=sum(2**arange(nsite-1,-1,-1)*asarray(config))
        filename='data/ket%s_h%.2fN%s.dat'%(ind,mean(hl),nsite)
        quicksave(filename,(eng,ket))
    overlap=abs((k0.tobra()*ket).item())
    print 'OVERLAP %s'%overlap
    return eng,ket,overlap

def random_config(nsite):
    config0=random.randint(0,2,nsite)
    return config0

def get_width(profile):
    '''Get the width from the ssf profile.'''
    nsite=len(profile)
    #get weighted average
    sites=arange(nsite)
    meansite=sum(profile*sites)
    sigma=sqrt(sum(profile*(sites-meansite)**2))
    return sigma

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

def solve_ed(hl,nsite,Jz=1.,J=1.):
    '''
    Run ED for Heisenberg model.
    '''
    #generate the model
    model=HeisenbergModel(J=J,Jz=Jz,h=hl,nsite=nsite)

    #run vmps
    #generate a random mps as initial vector
    spaceconfig=SpinSpaceConfig([2,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
    H=model.H_serial.H().real
    E,U=eigh(H)
    return E,U

def load_mps(hl,nsite,config):
    '''Load a MPS'''
    ind=sum(2**arange(nsite-1,-1,-1)*asarray(config))
    filename='data/ket%s_h%.2fN%s.dat'%(ind,mean(hl),nsite)
    return quickload(filename)

def ssf(mps1,mps2,direction='->'):
    '''
    SSF between two mpses.
    
    Return:
        tuple, (ssf, profile)
    '''
    if direction=='->':
        fl=asarray(SSFLR([mps1,mps2],direction='->'))
    else:
        fl=asarray(SSFLR([mps1,mps2],direction='<-'))[::-1]
    return fl,diff(fl**2)
