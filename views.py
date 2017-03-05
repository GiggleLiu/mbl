'''
View for Two Impurity Kondo model.
'''

from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import eigvalsh,eigh
import pdb,time

from tba.hgen import SpinSpaceConfig,quickload,quicksave
from tba.hgen.multithreading import RANK,COMM,SIZE
from dmrg import DMRGEngine,get_H_bm,get_H
from nrg.binner import Binner
from blockmatrix import eigbh
from models import HeisenbergModel,random_config,solve1,get_width,solve_ed,ssf
from pymps import get_expect,opunit_S

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

def show_ssf(hl,nsite,config1,config2):
    '''ssf between two configures.'''
    mps1=load_mps(hl,nsite,config1)[1]
    mps2=load_mps(hl,nsite,config2)[1]
    fl=ssf(mps1,mps2,'->')[0]
    fr=ssf(mps1,mps2,'<-')[0]
    sites=arange(nsite+1)
    ion()
    plot(sites,fl,color='k')
    plot(sites,fr,color='k')
    plot(sites[1:]-0.5,diff(fl),color='r')
    plot(sites[1:]-0.5,-diff(fr),color='b')
    legend([r'$[F^L]^2$',r'$[F^R]^2$'])
    pdb.set_trace()

def len_stat_E(h,nsite,nsample=1000):
    '''difference of characteristic length as a function of ds(E)'''
    datas=[]
    for i in xrange(nsample):
        #generate two random initial config with random site different.
        hl=(2*random.random(nsite)-1)*h
        config0=random_config(nsite)
        pos=random.randint(nsite)
        config1=copy(config0)
        config1[pos]=1-config0[pos]

        eng1,mps1,overlap1=solve1(hl,nsite,config=config0,Jz=1.,J=1.,save=False)
        eng2,mps2,overlap2=solve1(hl,nsite,config=config1,Jz=1.,J=1.,save=False)
        fl,profile=ssf(mps1,mps2,'->')
        sigma=get_width(profile)
        datas.append((pos,eng1,overlap1,eng2,overlap2,sigma))
    f=open('data/len_stat_h%.2f.dat'%h,'a')
    savetxt(f,datas)

def retip_old(h,nsite):
    '''difference of characteristic length as a function of ds(dE,dr)'''
    #generate a random config
    config0=random_config(nsite)
    E1L,E2L,SL,RL=[],[],[],[]
    for i in xrange(nsite):
        if i==nsite/2: continue
        config1=config0[:]
        config1[i]=1-config0[i]
        config2=config1[:]
        config2[nsite/2]=1-config2[nsite/2]
        eng1,mps1=solve1(h,nsite,config=config1,Jz=1.,J=1.,save=True)
        eng2,mps2=solve1(h,nsite,config=config2,Jz=1.,J=1.,save=True)
        #fr=(asarray(SSFLR([mps1,mps2],direction='<-'))**2)[::-1]
        fl=asarray(SSFLR([mps1,mps2],direction='->'))**2
        profile=diff(fl)
        sigma=get_width(profile)
        E1L.append(eng1)
        E2L.append(eng2)
        SL.append(sigma)
        RL.append(i-nsite/2)
    savetxt('data/')
    ion()
    plot(RL,SL)
    pdb.set_trace()

def measure_op(which,ket,usempi=False,ofile=None):
    '''
    Get order parameter from ground states.

    Parameters:
        :which: str, 

            * 'Sz', <Sz>
        :siteindices: list, the measure points.

    Return:
        array, the value of order parameter.
    '''
    nsite=ket.nsite
    spaceconfig=SpinSpaceConfig([2,1])
    ml=[]
    def measure1(siteindex):
        if which=='Sz':
            op=opunit_S(which='z',spaceconfig=spaceconfig,siteindex=siteindex)
        else:
            raise NotImplementedError()
        m=get_expect(op,ket=ket).item() #,bra=ket.tobra(labels=ket.labels))
        print 'Get %s for site %s = %s @RANK: %s'%(which,siteindex,m,RANK)
        return m

    if usempi:
        ml=mpido(measure1,inputlist=siteindices)
    else:
        ml=[]
        for siteindex in arange(nsite):
            ml.append(measure1(siteindex))
    ml=array(ml)
    if RANK==0 and ofile is not None:
        savetxt(ofile,ml.real)
    return ml

def show_sz(config,mps):
    '''show the '''
    ml1=measure_op('Sz',mps)
    plot(ml1)
    plot(-config+0.5)
    legend(['Sz','Config'])
    pdb.set_trace()
