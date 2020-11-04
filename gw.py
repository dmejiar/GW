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

def getxm(Pai, eai):
    RPA = 4.0*( Pai.T @ Pai )
    d = np.einsum('ii->i',RPA)
    d += eai
    AmB = np.sqrt(eai)
    RPA = np.einsum('i,ij,j->ij',AmB,RPA,AmB,optimize=True)
    Omega, RPA = linalg.eigh(RPA, check_finite=False, overwrite_a=True)
    Omega = np.sqrt(Omega)
    XPY = np.einsum("i,ij,j->ij",AmB,RPA,1.0/np.sqrt(Omega),optimize=True)
    Qs = Pai @ XPY
    return Qs.T, Omega

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
    Pai = Pai.reshape((naux,nomoa*numoa))
    epsilon_new = copy(gw_epsilon_a)
    linear = np.zeros(nqpa)
    pole = np.zeros(nqpa)

    Vqm = np.einsum('qmP,qmP->qm',qmP,qmP,optimize=True)
    Sigma_x = -np.sum(Vqm[:,:nomoa],axis=1)
    Sigma_new = Vxca + exx*Sigma_x
    eta2 = -9.0*np.real(ieta)**2
    for eviter in range(evcycles):
        print("")
        print("\t  QP       E_qp sol.      E_qp lin.         Z    ")
        print("\t-------------------------------------------------")
        epsilon_old = copy(epsilon_new)
        Sigma_old = copy(Sigma_new)
        if eviter == 0 or not evgw0:
           epsilon_ai = (epsilon_old[nomoa:,np.newaxis] - epsilon_old[np.newaxis,:nomoa]).reshape((nomoa*numoa))
           Qs, Omega = getxm(Pai, epsilon_ai)
        for qp in range(nqpa):
            epsilon_out = epsilon_old[ilowa+qp]
            broke = False
            #
            wns = 2.0*(Qs @ qmP[qp].T )**2
            #
            epsilon_in = copy(epsilon_out)
            omega = epsilon_in - epsilon_old
            Sigma_c = 0.0
            dSigma_c = 0.0
            for m in range(nomoa*numoa):
              temp = omega - Omega[m]*np.sign(epsilon_old)
              denom = 1.0/(temp**2 + eta2)
              factor = temp*denom
              dfactor = denom - 2*factor**2
              Sigma_c += factor @ wns[m]
              dSigma_c += dfactor @ wns[m]
            pole[qp] = max(0.0,min(1.0/(1.0 - dSigma_c),1.0))
            Sigma_new[qp] = Sigma_c + Sigma_x[qp]
            func = epsilon_old[ilowa+qp] + Sigma_new[qp] - Sigma_old[qp] - epsilon_in
            linear[qp] = epsilon_in + pole[qp]*func
            if epsilon_in*linear[qp] > 0.0:
              epsilon_new[ilowa+qp] = linear[qp]
              epsilon_in = copy(linear[qp])
            else:
              epsilon_new[ilowa+qp] += 0.1*np.sign(epsilon_in)
              epsilon_in += 0.1*np.sign(epsilon_in)
            #
            if pole[qp] > 0.7:
              nomegas = 25
              domega = 0.0005
            else:
              nomegas = 200
              domega = 0.0005
            #
            if epsilon_in < 0.0:
              grid = epsilon_in + np.linspace(-nomegas*domega,min(nomegas*domega,-epsilon_in),num=2*nomegas+1)
            else:
              grid = epsilon_in + np.linspace(max(-epsilon_in,-nomegas*domega),nomegas*domega,num=2*nomegas+1)

            omega = grid[:,np.newaxis] - epsilon_old[np.newaxis,: ]
            Sigma_c = np.zeros(2*nomegas+1)
            dSigma_c = np.zeros(2*nomegas+1)
            for m in range(nomoa*numoa):
              temp = omega - (Omega[m]*np.sign(epsilon_old))[np.newaxis,:]
              denom = 1.0/(temp**2 + eta2)
              factor = temp*denom
              dfactor = denom - 2*factor**2
              Sigma_c += factor @ wns[m]
              dSigma_c += dfactor @ wns[m]
            pole_grid = 1.0/(1.0 - dSigma_c)
            Sigma_grid = Sigma_c + Sigma_x[qp]
            rhs = epsilon_old[ilowa+qp] + Sigma_grid - Sigma_old[qp]
            g = rhs - grid
            pole[qp] = 0.0
            for i in range(2*nomegas):
               if g[i]*g[i+1] < 0.0:
                 z = max(pole_grid[i],pole_grid[i+1])
                 if z > pole[qp]:
                   pole[qp] = z
                   epsilon_new[ilowa+qp] = 0.5*(grid[i] + pole_grid[i]*g[i] + grid[i+1] + pole_grid[i+1]*g[i+1])
                   Sigma_new[qp] = (Sigma_grid[i] + Sigma_grid[i+1])/2.0
                   if z > 0.4: break
 
            
            #for iiter in range(15):
            #    epsilon_in = copy(epsilon_out)
            #    omega = epsilon_in - epsilon_old
            #    res = np.array([0.,0.,0.])
            #    for m in range(nomoa*numoa):
            #        temp = omega - Omega[m]*np.sign(epsilon_old)
            #        denom = 1.0/(temp**2 + eta2)
            #        factor = temp*denom
            #        dfactor = denom - 2*factor**2
            #        d2factor = -2.0*factor*(denom + 2.0*dfactor)
            #        temp2 = np.array([factor,dfactor,d2factor])
            #        res += temp2 @ wns[m]
            #    pole[qp] = 1.0/(1.0 - res[1])
            #    Sigma_new[qp] = res[0] + Sigma_x[qp]
            #    func = epsilon_old[ilowa+qp] + Sigma_new[qp] - Sigma_old[qp] - epsilon_in
            #    dfunc = res[1] - 1.0
            #    #
            #    func2 = func**2
            #    grad = 2.0*dfunc*func
            #    hess = 2.0*dfunc**2 + 2.0*func*res[2]
            #    step1 = -grad/hess
            #    step2 = pole[qp]*func
            #    if epsilon_in*(epsilon_in + step2) < 0.0:
            #      epsilon_out = 2*(2.0*epsilon_in + step2)/3.0
            #    else:
            #      epsilon_out = epsilon_in + step2
            #    if iiter == 0: linear[qp] = epsilon_in + step2
            #    epsilon_error = np.abs(epsilon_out - epsilon_in)*psi4.constants.hartree2ev
            #    if (epsilon_error < 0.01 and iiter >0) or dolinear: break
            #    iiter += 1
            #if iiter > 14: broke = True
            #epsilon_new[ilowa+qp] = epsilon_out
            #if broke: warning('QP {} energy did not converge.'.format(str(qp+1)))
            print(" {:>10d}       {:>8.2f}        {:>8.2f}     {:>8.3f}".format(qp+1,(epsilon_new[ilowa+qp]+scf_efermi_a)*psi4.constants.hartree2ev,(linear[qp]+scf_efermi_a)*psi4.constants.hartree2ev,pole[qp]))
        if evcycles > 1: print("\t evGW iteration {} complete".format(eviter+1))
    

