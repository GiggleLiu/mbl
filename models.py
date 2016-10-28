'''
Heisenberg Model.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
import pdb,time,copy

from tba.hgen import SpinSpaceConfig
from rglib.mps import MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,product_state,random_product_state
from rglib.hexpand import RGHGen
from rglib.hexpand import MaskedEvolutor,NullEvolutor,Evolutor
from dmrg import DMRGEngine
from blockmatrix import SimpleBMG

from dmrg import VMPSEngine

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
        mpo=mpc.toMPO(method='addition')
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
