#!/usr/bin/env python3

import sys
import os, errno

import psi4
import numpy as np
import argparse
from copy import deepcopy as copy
from scipy import linalg
from timeit import default_timer as timer

def warning(str):
    print("\t Warning: \t {}".format(str))
    return
    
def error(str):
    print("\t Error: \t {}".format(str))
    print("\t Aborting job")
    sys.exit(1)
    return

def removefile(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
    return


def buildpi(omega, Pai, eai):
    factor = np.sqrt(4*(ieta - eai)/((ieta - eai)**2 - omega**2))
    temp = np.einsum('ai,Pai->Pai',factor,Pai,optimize=True)
    PiPQ = -np.einsum('Pai,Qai->PQ', temp, temp, optimize=paiqai_pq_path)
    d = np.einsum('ii->i',PiPQ)
    d += 1.0
    return PiPQ

def buildw(PiPQ, qmP, Vqm):
    try:
        lower = linalg.cholesky(PiPQ, lower=True, check_finite=False)
        temp = linalg.solve_triangular(lower, qmP.T, lower=True, check_finite=False)
        temp = np.einsum('Pqm,Pqm->qm', temp, temp, optimize=qmpqmp_qm_path).T
    except:
        PiPQ = linalg.inv(PiPQ, check_finite=False)
        temp = np.einsum('PR,qmR->qmP', PiPQ, qmP, optimize=True)
        temp = np.einsum('qmP,qmP->qm', qmP, temp, optimize=True)
    return temp - Vqm



def parser():
    pars = argparse.ArgumentParser()
    pars.add_argument("xyz",help="xyz coordinate file")
    pars.add_argument("--dfa",help='exchange-correlation DFA (in Psi4 notation)',default='PBE')
    pars.add_argument("--etol",help='SCF energy tolerance',default=1.e-6,type=float)
    pars.add_argument("--obasis",help='orbital basis set',default='def2-tzvp')
    pars.add_argument("--abasis",help='auxiliary basis set type',default='RIFIT',choices=['RIFIT','JKFIT'])
    pars.add_argument("--memory",help='maximum amount of memory to be used in GB',default=2.0,type=float)
    pars.add_argument("--ngaussleg",help='number of points in the Gauss-Legendre quadrature',default=100,type=int)
    pars.add_argument("--no_qp",help='number of particle energies, starting from HOMO, to be computed',default=1,type=int)
    pars.add_argument("--nv_qp",help='number of hole energies to be computed',default=0,type=int)
    pars.add_argument("--eta",help="complex infinitesimal",default=0.001,type=float)
    pars.add_argument("--verbosity",help="set verbosity level",default=0,type=int)
    pars.add_argument("--nthreads",help="number of MKL threads",default=1,type=int)
    pars.add_argument("--dftgrid",help="preset density of the numerical integration grid",default='medium',choices=['coarse','medium','fine'])
    pars.add_argument("--evcycles",default=1,type=int,help='Request an evGW calculation with EVCYCLES number of cycles')
    pars.add_argument("--eps",default=0.00001,type=float,help='Energies within EPS from each other will be considered degenerate')
    pars.add_argument("--fdstep",default=0.001,type=float,help='Step size for the finite-difference derivatives')
    pars.add_argument("--linear",action='store_true',help='If present, use the linearized solution. Otherwise sove the quasiparticle equation')
    pars.add_argument("--evgw0",action='store_true',help='Keep W0 fixed for the evGW_0 scheme. Needs EVCYCLES >1')
    return pars.parse_args()

def header():
    print("="*56)
    print("""
                  ######   ##      ## 
                 ##    ##  ##  ##  ## 
                 ##        ##  ##  ## 
                 ##   #### ##  ##  ## 
                 ##    ##  ##  ##  ## 
                 ##    ##  ##  ##  ## 
                  ######    ###  ###  
    """)
    print("="*56)
    print("")
    

def readin():
    global noqpa, nuqpa, ieta, ngrid, abasis, evcycles, eps, fdstep, dft_functional, dolinear, evgw0, xyz
    
    args = vars(parser())
    if not os.path.isfile(args['xyz']): error("File '{}' does not exist!".format(args.xyz))
    if args['etol'] <= 0.0: error("--etol must be > 0.0")
    if args['memory'] <= 0.0: error("--memory must be > 0.0")
    if args['ngaussleg'] < 1: error("--ngausleg must be > 0")
    if args['no_qp'] < -1: error("--no_qp must be >= -1")
    if args['nv_qp'] < -1: error("--nv_qp must be >= -1")
    if args['no_qp'] == 0 and args['nv_qp'] == 0: error("No quasiparticles!")
    if args['eta'] <= 0.0: error("--eta must be > 0.0")
    if args['nthreads'] < 1: error("--nthreads must be > 0")
    if args['evcycles'] < 1: error("--evcycles must be > 0")
    if args['fdstep'] <= 0.0: error("--fdstep must be > 0.0")
    if args['eps'] < 0.0: error("--eps must be >= 0.0")
    
    noqpa = args['no_qp']
    nuqpa = args['nv_qp']
    ieta  = 1j*args['eta']
    ngrid = args['ngaussleg']
    abasis = args['abasis']
    dfa = args['dfa']
    evcycles = args['evcycles']
    eps = args['eps']
    fdstep = args['fdstep']
    dolinear = args['linear']
    evgw0 = args['evgw0']
    xyz = args['xyz']

    if dfa in ['scan', 'SCAN']:
        dft_functional = {"name": 'scan', 
             "x_functionals": {"MGGA_X_SCAN": {}},
             "c_functionals": {"MGGA_C_SCAN": {}}
        }
    else:
        dft_functional = dfa

    if evcycles > 1:
        noqpa = -1; nuqpa = -1
     
    ### Define XC grid parameters
    if args['dftgrid'] == 'coarse':
        nradial = 50; nangular = 266
    elif args['dftgrid'] == 'fine':
        nradial = 150; nangular = 590
    else:
        nradial = 100; nangular = 350

    ### Set Psi4 options
    
    if 'MKL_NUM_THREADS' in os.environ:
        if args['nthreads'] != int(os.environ['MKL_NUM_THREADS']):
            warning("nthreads != MKL_NUM_THREADS. Using MKL_NUM_THREADS")
            args['nthreads'] = int(os.environ['MKL_NUM_THREADS'])
    try:
        import mkl
        mkl.set_num_threads(args.nthreads)
    except:
        pass
    psi4.core.set_num_threads(args['nthreads'])
    psi4.core.set_output_file('__psi4output.dat')
    psi4.set_memory(args['memory']*1.0e9)
    psi4.set_options({'basis': args['obasis'], 'e_convergence': args['etol'], 'scf_type': 'df', 'dft_spherical_points': nangular, 'dft_radial_points': nradial})
    #
    molstr = open(args['xyz'],'r').readlines()
    molstr.append("    symmetry c1\n    units angstrom")
    mol = ''.join(molstr[2:])
    lmol = psi4.geometry("""
    {}""".format(mol))
    #
    print("")
    return lmol

def readwfn(wfn):
    global nbf,nomoa,nomob,numoa,numob,noqpa,noqpb,nuqpa,nuqpb,obasis
    global nqpa, ilowa, iuppa, scf_efermi_a, scf_epsilon_a, gw_epsilon_a, Vxca
    global Ca, Cqpa, exx
    # 
    d1 = timer()
    print("\t Data from Wavefunction object ...           ",end="")
    
    # basis set object
    obasis = wfn.basisset()

    # number of basis functions, occupied and virtual orbitals
    nbf = wfn.nmo()
    nomoa = wfn.nalpha()
    nomob = wfn.nbeta()
    numoa = nbf - nomoa
    numob = nbf - nomob
    if nomoa != nomob: error("\t Open-shel case is not available yet")

    # number of quasiparticle energies to compute
    noqpa = nomoa if noqpa == -1 else noqpa
    nuqpa = numoa if nuqpa == -1 else nuqpa
    nqpa = noqpa + nuqpa  
    ilowa = nomoa - noqpa
    iuppa = nomoa + nuqpa

    # Molecular orbital coefficients and eigenvalues
    Ca = np.array(wfn.Ca())
    Cqpa = Ca[:,ilowa:iuppa]
    scf_epsilon_a = np.array(wfn.epsilon_a())
    scf_efermi_a  = (scf_epsilon_a[nomoa] + scf_epsilon_a[nomoa-1])/2

    # Initialize guess
    gw_epsilon_a = scf_epsilon_a - scf_efermi_a

    # Read and transform Vxc matrix
    Vxca = np.array(wfn.Va())
    Vxca = Vxca @ Cqpa
    Vxca = np.diag(Cqpa.T @ Vxca)
    exx = wfn.V_potential().functional().x_alpha() 

    print("\t {:6.2f} seconds".format(timer()-d1))
    return
 

def eris(C):
    global naux
    # 
    aux = psi4.core.BasisSet.build(lmol, "DF_BASIS_SCF", "", abasis, obasis.name())
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(obasis)
    #
    d1 = timer()
    print("\t Two-center ERIs ...                         ",end="")
    VPQ = np.array(mints.ao_eri(zero,aux,zero,aux))[0,:,0,:]
    VPQ = linalg.cholesky(VPQ, lower=True, check_finite=False)
    print("\t {:6.2f} seconds".format(timer()-d1))
    #
    d1 = timer()
    print("\t Three-center ERIs in AO representation ...  ",end="")
    Pnm = np.array(mints.ao_eri(zero, aux, obasis, obasis))[0,:,:,:]
    print("\t {:6.2f} seconds".format(timer()-d1))
    #
    d1 = timer()
    print("\t Three-center ERIs in MO representation ...  ",end="")
    Pnm = Pnm @ C[:,:iuppa]
    Pnm = C.T @ Pnm
    print("\t {:6.2f} seconds".format(timer()-d1))
    #
    d1 = timer()
    print("\t Solution of triangular problem ...          ",end="")
    Pnm = linalg.solve_triangular(VPQ, Pnm, lower=True, check_finite=False)
    print("\t {:6.2f} seconds".format(timer()-d1))
    #
    Pai = Pnm[:,nomoa:,:nomoa].copy()
    qmP = Pnm[:,:,ilowa:iuppa].T.copy()
    naux = len(VPQ)
    #
    return Pai, qmP
     

def gaussleg():
    global wp, w2, gl_w
    d1 = timer()
    print("\t Generate Gauss-Legendre grid ...            ",end="")
    gl_x, gl_w = np.polynomial.legendre.leggauss(ngrid)
    gl_w = gl_w / (1 - gl_x)**2 / np.pi
    wp = (1 + gl_x) / (1 - gl_x)
    w2 = (wp**2)[:,np.newaxis]
    print("\t {:6.2f} seconds".format(timer()-d1))
    return 

def buildiw(Pai, epsilon_ai, Vqm):
    iw = 1j*wp
    IWqm = np.zeros((ngrid,nqpa,nbf), dtype='complex128')
    #
    for idx in range(ngrid):
        PiPQ = buildpi(iw[idx], Pai, epsilon_ai)
        IWqm[idx] = buildw(PiPQ, qmP, Vqm)
    #
    for qp in range(nqpa):
        filename = "IWqm_qp_"+str(qp)+".npy"
        removefile(filename)
        np.save(filename,IWqm[:,qp,:])
    return

def buildin(epsilon_in, epsilon, qp):
    factor = epsilon_in - epsilon + ieta*np.sign(epsilon)
    factor2 = factor**2
    IW = np.einsum('gm,g->gm',np.load("IWqm_qp_"+str(qp)+".npy"),gl_w,optimize=True)
    km = 2*factor[np.newaxis,:]/(factor2[np.newaxis,:] + w2)
    In = np.einsum('gm,gm',IW,km,optimize=True)
    return In

def buildrn(epsilon_in, epsilon, ai, Pai, qmP, Vqm):
    Rn = 0.0j
    for ibf in range(nbf):
        if epsilon[ibf] > 0.0:
          if epsilon[ibf] > epsilon_in + eps: continue
          f = 0.5 if np.isclose(epsilon[ibf],epsilon_in,eps) else 1.0
          omega = epsilon[ibf] - epsilon_in - ieta
        elif epsilon[ibf] < 0.0:
          if epsilon[ibf] < epsilon_in - eps: continue
          f = -0.5 if np.isclose(epsilon[ibf],epsilon_in,eps) else -1.0
          omega = epsilon[ibf] - epsilon_in + ieta
        PiPQ = buildpi(omega, Pai, ai)
        vector = linalg.solve(PiPQ,  qmP[ibf], check_finite=False, assume_a='sym', overwrite_a=True)
        Rn += f*(np.dot(vector,qmP[ibf]) - Vqm[ibf])
    return Rn



def dogw(epsilon, ai, Sigma_old, Vqm, qmP, Pai, Sigma_x, qp):
    broke = False
    iiter = 0
    epsilon_out = copy(epsilon[ilowa+qp])
    while iiter < 5:
        epsilon_in = copy(epsilon_out)
        #
        In = buildin(epsilon_in, epsilon, qp)
        Inp = buildin(epsilon_in+fdstep, epsilon, qp)
        Inm = buildin(epsilon_in-fdstep, epsilon, qp)
        #
        Rn = buildrn(epsilon_in, epsilon, ai, Pai, qmP, Vqm)
        Rnp = buildrn(epsilon_in+fdstep, epsilon, ai, Pai, qmP, Vqm)
        Rnm = buildrn(epsilon_in-fdstep, epsilon, ai, Pai, qmP, Vqm)
        #
        Sigma_c = Rn - In
        dSigma_c = (Rnp - Rnm - Inp + Inm)/(2*fdstep)
        d2Sigma_c = (Rnp - 2*Rn + Rnm - Inp + 2*In - Inm)/fdstep**2
        #
        Sigma_new = np.real(Sigma_c + Sigma_x)
        #
        Zn = np.real(1/(1 - dSigma_c))
        #Zn = max(Zn,0.01)
        #Zn = min(Zn,1.0)
        #
        func = epsilon[ilowa+qp] + Sigma_new - Sigma_old - epsilon_in
        dfunc = np.real(dSigma_c - 1.0)
        #
        func2 = func**2
        grad = 2.0*dfunc*func
        hess = 2.0*dfunc**2 + 2.0*func*np.real(d2Sigma_c)
        #
        #if iiter>0 and np.real(hess) > 0.1:
        #  epsilon_out = epsilon_in - np.real(grad/hess)
        #else:
        if np.abs(Zn) < 0.001:
          epsilon_out = epsilon_in - grad/hess
        else: 
          epsilon_out = epsilon_in + Zn*func
        if iiter == 0: epsilon_linear = epsilon_in + Zn*func
        #print("\t\t {} {}".format(iiter,epsilon_out))
        #
        epsilon_error = np.abs(epsilon_out - epsilon_in)*psi4.constants.hartree2ev
        #
        if epsilon_error < 0.01 or dolinear: break
        iiter += 1
    if iiter == 20: broke = True
    return epsilon_out, epsilon_linear, Sigma_new, Zn, broke




if __name__ == '__main__':
    header()
    lmol = readin()
    #
    d1 = timer()
    scf, wfn = psi4.energy('SCF', dft_functional=dft_functional, return_wfn=True)
    print("\t SCF step done in {:6.2f} seconds.\n".format(timer()-d1))
    #
    readwfn(wfn)
    del wfn
    #
    Pai, qmP = eris(Ca)
    gaussleg()
    #
    epsilon_ai = scf_epsilon_a[nomoa:,np.newaxis] - scf_epsilon_a[np.newaxis,:nomoa]
    epsilon_new = copy(gw_epsilon_a)
    linear = np.zeros(nqpa)
    pole = np.zeros(nqpa)
    paiqai_pq_path = np.einsum_path('Pai,Qai',Pai,Pai,optimize='optimal')[0]
    pqmpqm_qm_path = np.einsum_path('Pqm,Pqm->qm',qmP,qmP,optimize='optimal')[0]

    Vqm = np.einsum('qmP,qmP->qm',qmP,qmP,optimize=True)
    Sigma_x = -np.sum(Vqm[:,:nomoa],axis=1)
    Sigma_new = Vxca + exx*Sigma_x
    for eviter in range(evcycles):
        print("")
        print("\t  QP       E_qp sol.      E_qp lin.         Z    ")
        print("\t-------------------------------------------------")
        epsilon_old = copy(epsilon_new)
        Sigma_old = copy(Sigma_new)
        if eviter == 0 or not evgw0:
           epsilon_ai = epsilon_old[nomoa:,np.newaxis] - epsilon_old[np.newaxis,:nomoa]
           buildiw(Pai, epsilon_ai, Vqm)
        for qp in range(nqpa):
            epsilon_new[qp], linear[qp], Sigma_new[qp], pole[qp], broke = dogw(epsilon_old, epsilon_ai, Sigma_old[qp], Vqm[qp], qmP[qp], Pai, Sigma_x[qp],qp)
            if broke: warning('QP {} energy did not converge.'.format(str(qp+1)))
            print(" {:>10d}       {:>8.2f}        {:>8.2f}     {:>8.3f}".format(qp+1,(epsilon_new[qp]+scf_efermi_a)*psi4.constants.hartree2ev,(linear[qp]+scf_efermi_a)*psi4.constants.hartree2ev,pole[qp]))
        epsilon_error = epsilon_new - epsilon_old
        if evcycles > 1: print("\t evGW iteration {} complete".format(eviter+1))
        print("")
    for qp in range(nqpa):
        filename = "IWqm_qp_"+str(qp)+".npy"
        removefile(filename)
    with open('GW100.out', 'a') as f:
        f.write("{}: {:8.3f}    {:8.3f}\n".format(xyz, (epsilon_new[0]+scf_efermi_a)*psi4.constants.hartree2ev, (linear[0]+scf_efermi_a)*psi4.constants.hartree2ev))

    

