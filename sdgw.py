#!/usr/bin/env python3

import sys
import os
import errno
import psi4
import argparse
import numpy as np
from math import isclose
from copy import deepcopy as copy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres as MINRES
from timeit import default_timer as timer

global spdict       # Translate iangular value to a valid dft_spherical_points value
global ha2ev        # Hartree to eV conversion factor

ha2ev = 27.2114


spdict = {1: 6,     2: 14,    3: 26,    4: 38,    5: 50,    6: 74,
          7: 86,    8: 110,   9: 146,   10: 170,  11: 194,  12: 230,
          13: 266,  14: 302,  15: 350,  16: 434,  17: 590,  18: 770,
          19: 974,  20: 1202, 21: 1454, 22: 1730, 23: 2030, 24: 2354,
          25: 2702, 26: 3074, 27: 3470, 28: 3890, 29: 4334, 30: 4802,
          31: 5294, 32: 5810}
      

    
def error(string):
    print("\t Error: \t {}".format(string))
    print("\t Aborting job")
    sys.exit(1)
    return



def parser():
    """
    Input parser.
    
    Options related to Psi4:
        xcfun
        scftol
        aobasis
        cdbasis
        iangular
        nradial
        mult
        charge
        
    Options related to GW
        ngl
        noqp{a/b}
        nvqp{a/b}
        ieta
        evgw
        evgw0
        core
        maxnewton
        minres
        debug
    """
        
    # Input parser
    pars = argparse.ArgumentParser()
    pars.add_argument("xyz", help="Path to XYZ coordinate file")
    pars.add_argument("--xcfun", help="Name of XC functional to be used in GS calculation", default="PBE")
    pars.add_argument("--scftol", help="SCF energy tolerance", type=float, default=1.0E-7)
    pars.add_argument("--aobasis", help="Name of AO basis set", default="def2-tzvp")
    pars.add_argument("--cdbasis", help="Name of CD basis set", default="def2-tzvp-ri")
    pars.add_argument("--noqpa", help="Number of Occupied QP energies ALPHA spin", type=int, default=1)
    pars.add_argument("--noqpb", help="Number of Occupied QP energies BETA spin", type=int, default=1)
    pars.add_argument("--nvqpa", help="Number of Virtual QP energies ALPHA spin",type=int, default=0)
    pars.add_argument("--nvqpb", help="Number of Virtual QP energies BETA spin", type=int, default=0)
    pars.add_argument("--ieta", help="Imaginary infinitesimal value", type=float, default=0.01)
    pars.add_argument("--nthreads", help="Number of MKL threads", default=1, type=int)
    pars.add_argument("--evgw", help="Do an evGW self-consistent calculation", action="store_true")
    pars.add_argument("--evgw0", help="Do an evGW_0 self-consistent calculation", action="store_true")
    pars.add_argument("--core", help="If true, start counting from the core", action="store_true")
    pars.add_argument("--maxnewton", help="Maximum number of Newton steps per QP", type=int, default=15)
    pars.add_argument("--maxev", help="Maximum number of evGW or evGW_0 cycles", type=int, default=0)
    pars.add_argument("--memory", help="Maximum memory argument, in GB, passed to Psi4", type=float, default=1.0)
    pars.add_argument("--iangular", help="Angular grid quality index", type=int, default=13)
    pars.add_argument("--nradial", help="Number of radial shells", type=int, default=100)
    pars.add_argument("--mult", help="Multiplicity of the system", type=int, default=1)
    pars.add_argument("--charge", help="Charge of the system", type=int, default=0)
    pars.add_argument("--debug", help="Print debug information", action="store_true")
    return vars(pars.parse_args())



def runpsi4(args):
    """
    Run Psi4 and extract all relevant information and store
    it in global variables
    """
    
    global aobasis, cdbasis
    global movecs, evals, vxc
    global nocc, nvir, nmo
    global ipol, exx
    global efermi
    
    print(args['xyz'])
    # Read XYZ file
    try:
        molstr = open(args['xyz'],'r').readlines()
    except:
        error("Could not open '{}'".format(args['xyz']))
    
    # Get Psi4 molecule object
    molstr.append("    symmetry c1\n    units angstrom")
    mol = psi4.geometry("""
    {} {}
    {}""".format(args['charge'],args['mult'],''.join(molstr[2:])))
    
    # Set MKL threading
    try:
        import mkl
        mkl.set_num_threads(args['nthreads'])
    except:
        pass

    # Define Spin-polarization index (1: Unpolarized, 2: Polarized)
    ipol = 2 if args['mult'] > 1 else 1

    # Special DFT functionals
    if args['xcfun'].lower() == 'scan':
        dft_functional = {"name": "scan", "x_functionals": {"MGGA_X_SCAN": {}}, "c_functionals": {"MGGA_C_SCAN": {}}}
    elif args['xcfun'].lower() == 'r2scan':
        dft_functional = {"name": "r2scan", "x_functionals": {"MGGA_X_R2SCAN": {}}, "c_functionals": {"MGGA_C_R2SCAN": {}}}
    elif args['xcfun'].lower()[0:4] == 'pbeh':
        if len(args['xcfun']) > 4:
            exx = min(int(args['xcfun'][4:])/100.0,1.0)
        else:
            exx = 0.45
        dft_functional = {"name": "pbealpha", "x_functionals": {"GGA_X_PBE": {"alpha": (1.0-exx)}}, "x_hf": {"alpha": exx}, "c_functionals": {"GGA_C_PBE": {}}}
    else:
        dft_functional = args['xcfun']
    
    # Set other Psi4 options
    psi4.core.set_num_threads(args['nthreads'])
    psi4.core.set_output_file('__psi4output.dat')
    psi4.set_memory(args['memory']*1.0E9)
    psi4.set_options({
        'basis': args['aobasis'],
        'e_convergence': args['scftol'],
        'scf_type': 'df',
        'dft_spherical_points': spdict[args['iangular']],
        'dft_radial_points': args['nradial']})
    
    # Run Psi4
    scf, wfn = psi4.energy('SCF', dft_functional=dft_functional, return_wfn=True)
    
    # Read wavefunction object
    aobasis = wfn.basisset()
    cdbasis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", args['cdbasis'])
    nmo = wfn.nmo()
    nocc = [wfn.nalpha(), wfn.nbeta()]
    nvir = [nmo - occ for occ in nocc]
    
    movecs = [np.array(wfn.Ca())]
    evals  = [np.array(wfn.epsilon_a())]
    vxc    = [np.array(wfn.Va())]
    efermi = [0.5*(evals[0][nocc[0]] + evals[0][nocc[0]-1])]
    evals[0] -= efermi[0]

    if ipol > 1:
        movecs.append(np.array(wfn.Cb()))
        evals.append(np.array(wfn.epsilon_b()))
        vxc.append(np.array(wfn.Vb()))
        efermi.append(0.5*(evals[1][nocc[1]] + evals[1][nocc[1]-1]))
        evals[1] -= efermi[1]
        
    exx = wfn.V_potential().functional().x_alpha()
    
    del wfn, scf

    return




def integrals(args):
    """
    Compute 2-center (metric) and 3-center 2-body integrals in AO basis.
    The 3-center integrals are transformed to the MO basis.
    The resulting MO integrals are stored in a global variable.
    """
    
    global Prs, Pia, naux
    
    # Zero-basis for density fitting
    zerobas     = psi4.core.BasisSet.zero_ao_basis_set()
    
    # Psi4 integral Helper
    ints_helper = psi4.core.MintsHelper(aobasis)

    # Compute two-center integral matrix and perform Cholesky-decomposition
    d1 = timer()
    print("\t Two-center integrals ...               ", end="")
    VPQ = np.array(ints_helper.ao_eri(cdbasis,zerobas,cdbasis,zerobas)).squeeze()
    VPQ = linalg.cholesky(VPQ, lower=True, check_finite=False)
    naux = len(VPQ)
    print("\t {:8.2f} seconds".format(timer()-d1))
    
    # Obtain 3-center integrals in AO representation
    d1 = timer()
    print("\t Three-center integrals in AO basis ... ", end="")
    Pmn = np.array(ints_helper.ao_eri(cdbasis,zerobas,aobasis,aobasis)).squeeze()
    print("\t {:8.2f} seconds".format(timer()-d1))
    
    # Transform 3-center integrals to MO representation
    d1 = timer()
    print("\t Three-center integrals in MO basis ... ", end="")
    
    Prs = [np.einsum('pmn,mi,nr->pir',Pmn,movecs[0][:,:hi[0]],movecs[0],optimize=True)]
    if ipol > 1:
        Prs.append(np.einsum('pmn,mi,nr->pir',Pmn,movecs[1][:,:hi[1]],movecs[1],optimize=True))
    print("\t {:8.2f} seconds".format(timer()-d1))
    
    # Orthonormalize cdbasis with the Cholesky factors
    d1 = timer()
    print("\t Orthonormalize charge-density basis ... ", end="")
    for ispin in range(ipol):
        Prs[ispin] = linalg.solve_triangular(VPQ, Prs[ispin], lower=True, check_finite=False)
    
    # Pointer to Occ-Vir block
    Pia = []
    Pia.append(Prs[0][:,:nocc[0],nocc[0]:])     # Pointer to the Occ-Vir block
    if ipol > 1:
        Prs[1] = linalg.solve_triangular(VPQ, Prs[1], lower=True, check_finite=False)
        Pia.append(Prs[1][:,:nocc[1],nocc[1]:])     # Pointer to Occ-Vir block

    print("\t {:8.2f} seconds".format(timer()-d1))
    
    del VPQ, Pmn
    
    return



def gw_pars(args):
    """
    Setup some variables related to the GW calculation
    and store them in global variables
    """
    global docore, evgw, evgw0
    global noqp, nvqp, nqp
    global lo, hi
    global ngrid, maxev
    global minres, debug
   
    if ipol > 1:
        error("Open-shell case not implemented")
    
    docore = args['core']
    evgw = args['evgw']
    evgw0 = args['evgw0']
    
    noqp = [0, 0]; nvqp = [0, 0]; lo = [0, 0]; hi = [0, 0]; nqp = [0, 0]
    
    for ispin in range(ipol):
        string = 'a' if ispin == 0 else 'b'
        noqp[ispin] = nocc[ispin] if (args['noqp'+string] < 0 or evgw or evgw0) else args['noqp'+string]
        nvqp[ispin] = nmo - nocc[ispin] if args['nvqp'+string] < 0 else args['nvqp'+string]
    
        if docore:
            if nvqp[ispin] > 0 and noqp[ispin] < nocc[ispin]:
                print("\t Warning: nvqp{1} > 0 and noqp{1} < nocc is incompatible with --core".format(string))
                print("\t          setting nvqp{} to 0".format(string))
                nvqp[ispin] = 0
            lo[ispin] = 0
            hi[ispin] = max(noqp[ispin] + nvqp[ispin], nocc[ispin])
        else:
            lo[ispin] = nocc[ispin] - noqp[ispin]
            hi[ispin] = nocc[ispin] + nvqp[ispin]
        nqp[ispin] = noqp[ispin] + nvqp[ispin]
        
    maxev = max(args['maxev']+1,4) if (evgw or evgw0) else args['maxev']+1
    debug = args['debug']
    

    
def print_iter(inewton,ein,eout,lower,upper,bracket):
    print("\t Iter: {} Ein: {:12.6f} Eout: {:12.6f}".format(inewton,ein*ha2ev,eout*ha2ev))
    print("\t          Res: {:12.6f} Step: {:12.6f}".format(ha2ev*residual,ha2ev*(eout-ein)),end="")
    if bracket:
        print(" lower: {:12.6f} upper: {:12.6f}".format(ha2ev*elower,ha2ev*eupper))
    else:
        print("")
    return
    
    
def findclusters(vals,nqp):
    """
    Finds tightly clustered eigenvalues. For a given cluster,
    the starting QP energies are guessed from the previous
    solution inside the cluster.
    
    If HOMO and LUMO are close together, they are split in
    different clusters.
    """
    nclusters = 0
    clusters = []
    ll = 0
    
    while True:
        nclusters += 1
        icluster = 1
        target = vals[ll] + 0.05
    
        for iqp in range(ll+1,nqp):
            if vals[iqp]*vals[iqp-1] < 0.0:
                break
            if vals[iqp] <= target:
                icluster += 1
            else:
                break
        clusters.append(icluster)
        ll += icluster
    
        if ll >= nqp:
            break
        
    return clusters



def scissor(old,new,noqp,nvqp,nomo,ilow,iupp,spin):
    """
    Apply the Scissor operator to the remaining states
    in both the occupied and virtual spectrums
    """
    if spin not in ['Alpha','Beta']:
        error('Unrecognized spin {}'.format(spin))
    
    # Occupied states
    if noqp < nomo and noqp > 0:
        # Get average shift
        shift = 0.0
        for iqp in range(ilow,min(iupp,nomo)):
            shift += new[iqp] - old[iqp]
        shift /= noqp
        
        print("\t Applying {:8.4f} eV shift to rest of {} occupied states\n".format(shift*ha2ev,spin))
        
        # Apply shift
        for iqp in range(nomo):
            if iqp >= ilow and iqp < iupp:
                continue
            new[iqp] = old[iqp] + shift
    
    # Virtual states
    if nvqp < nmo-nomo and nvqp > 0:
        # Get average shift
        shift = 0.0
        for iqp in range(nomo,nomo+nvqp):
            shift += new[iqp] - old[iqp]
        shift /= nvqp
        
        print("\t Applying {:8.4f} eV shift to rest of {} virtual states\n".format(shift*ha2ev,spin))
        
        # Apply shift
        for iqp in range(nomo+nvqp,nmo):
            new[iqp] = old[iqp] + shift
    
    return
            

    
def getxm(_Pia, wia):
    s1 = _Pia.shape[1]
    s2 = _Pia.shape[2]
    size = s1*s2
    RPA = 4.0*np.einsum('Pia,Pjb->iajb',_Pia,_Pia,optimize=True).reshape((size,size))
    d = np.einsum('ii->i',RPA)
    d += wia.reshape(size)
    AmB = np.sqrt(wia.reshape(size))
    RPA = np.einsum('i,ij,j->ij',AmB,RPA,AmB,optimize=True)
    Omega, RPA = linalg.eigh(RPA, check_finite=False)
    Omega = np.sqrt(Omega)
    XPY = np.einsum("i,ij,j->ij",AmB,RPA,1.0/np.sqrt(Omega),optimize=True)
    Qs = np.einsum('Pi,ij->Pj',_Pia.reshape((naux,size)),XPY,optimize=True)
    return Qs.reshape((naux,s1,s2)), Omega.reshape((s1,s2))


def getwmn(_Prs, Qs, iqp):
    wmn = np.einsum('Pn,Pia->nia',_Prs[:,iqp,:],Qs,optimize=True)
    wmn = 2.0*np.einsum('nia,nia->nia',wmn,wmn,optimize=True)
    return wmn

def getsigmac(wmn, Omega, omega, vals):
    sigmac = 0.0; dsigmac = 0.0
    for imo in range(nmo):
        sgn = -np.sign(vals[imo])
        temp = omega - vals[imo] + sgn*Omega
        denom = 1.0/(temp**2 + 9.0*args['ieta']**2)
        sigmac += np.einsum('ia,ia,ia->',wmn[imo],temp,denom,optimize=True)
        temp = args['ieta']**2 - temp**2
        dsigmac += np.einsum('ia,ia,ia,ia->',wmn[imo],temp,denom,denom,optimize=True)
    return sigmac, dsigmac



if __name__ == '__main__':
    global factor_a, factor_b
    
    # Parse the command line
    args = parser()
    
    # Run Psi4
    runpsi4(args)

    # Initialize GW parameters
    gw_pars(args)
    
    # Generate integrals
    integrals(args)
    
    # Transform VXC matrices to MO basis (just diagonal)
    vxc[0] = np.einsum('ij,jk->ik', vxc[0], movecs[0][:,lo[0]:hi[0]])
    vxc[0] = np.einsum('ij,ij->j', movecs[0][:,lo[0]:hi[0]], vxc[0])
    if ipol > 1:
        vxc[1] = np.einsum('ij,jk->ik', vxc[1], movecs[1][:,lo[1]:hi[1]])
        vxc[1] = np.einsum('ij,ij->j', movecs[1][:,lo[1]:hi[1]], vxc[1])
        
    # Bare Coulomb super-diagonal
    Vmn = [np.einsum('Pij,Pij->ij',Prs[0],Prs[0],optimize=True)]
    if ipol > 1:
        Vmn.append(np.einsum('Pij,Pij->ij',Prs[1],Prs[1],optimize=True))
    
    # Sigma_x
    Sigmax = [-np.einsum('ij->i',Vmn[0][lo[0]:hi[0],:nocc[0]])]
    if ipol > 1:
        Sigmax.append(-np.einsum('ij->i',Vmn[1][lo[1]:hi[1],:nocc[1]]))
        
    ######################################
    #### START evGW/evGW_0 iterations ####
    ######################################
    
    newevals = copy(evals)
                    
    for eviter in range(maxev):
        if evgw:
            print("\n\t G{0}W{0}".format(eviter))
        elif evgw0:
            print("\n\t G{}W0".format(eviter))
        else:
            print("\n\t G0W0")
        
        
        # Calculate eval difference in first iteration or in evGW
        if eviter == 0 or evgw:
            wia = [newevals[0][np.newaxis,nocc[0]:] - newevals[0][:nocc[0],np.newaxis]]
            if ipol > 1:
                wia.append(newevals[1][np.newaxis,nocc[1]:] - newevals[1][:nocc[1],np.newaxis])
        
        # Copy evals
        oldevals = copy(newevals)
            
        # Calculate RPA Polarizability
        if eviter == 0 or evgw:
            Qs, Omega = getxm(Pia[0],wia[0])
        
        # Loop over spin channels
        for ispin in range(ipol):
            
            if ispin == 0:
                string= "Alpha Orbitals"
            else:
                string="Beta Orbitals"
               
            # Quick return for no active QPs
            if nqp[ispin] < 1:
                continue
            
            print("\t                 {}             ".format(string))
            print("\t      State      Energy (eV)      Error (eV)  ")
            print("\t      --------------------------------------  ")
            
            warning = False
            fixed = [False]*nqp[ispin]
            esterror = np.zeros(nqp[ispin])
            
            # Find clusters of eigenvalues
            clusters = findclusters(oldevals[ispin][lo[ispin]:],nqp[ispin])
                
            # Loop over all clusters of eigenvalues
            ulqp = -1
            for icluster in range(len(clusters)):
                    
                # Define limits for current cluster
                llqp = ulqp + 1
                ulqp = ulqp + clusters[icluster]
            
                mylo = llqp + 0
                myhi = ulqp + 0
                
                while True:
                    
                    # For occupied states, start from upper to lower
                    # For virtual states, start from lower to upper
                    if lo[ispin]+llqp < nocc[ispin]:
                        iqp = myhi
                    else:
                        iqp = mylo
                    
                    _eout = oldevals[ispin][lo[ispin]+iqp]
                    
                    # Guess energy from previous QP
                    if eviter < 2:
                        if myhi < ulqp:
                            _eout = newevals[ispin][lo[ispin]+iqp+1] + 0
                        elif mylo > llqp:
                            _eout = newevals[ispin][lo[ispin]+iqp-1] + 0
                        
                    # Get Wmn
                    wmn = getwmn(Prs[ispin],Qs,iqp)

                    eupper = 1.0
                    elower = 0.0
                    values = np.zeros(args['maxnewton'])
                    errors = np.zeros(args['maxnewton'])
                    constant = evals[ispin][lo[ispin]+iqp] - vxc[ispin][iqp] + (1.0 - exx)*Sigmax[ispin][iqp]
                    bracket = False
                    
                    # Start Newton iteration
                    for inewton in range(args['maxnewton']):
                        
                        _ein = _eout + 0
                        
                        # Get Sigma_c
                        _sigmac, _dsigmac = getsigmac(wmn, Omega, _ein, oldevals[ispin])
                                                        
                        residual = _sigmac - _ein + constant
                        dresidual = _dsigmac - 1.0
                        
                        values[inewton] = _ein
                        errors[inewton] = residual
                        
                        # Check if we have bracketed the solution
                        if not bracket and inewton > 0:
                            if errors[inewton]*errors[inewton-1] < 0.0:
                                bracket = True
                                if values[inewton] > values[inewton-1]:
                                    elower = values[inewton-1]
                                    eupper = values[inewton]
                                    rlower = errors[inewton-1]
                                    rupper = errors[inewton]
                                else:
                                    elower = values[inewton]
                                    eupper = values[inewton-1]
                                    rlower = errors[inewton]
                                    rupper = errors[inewton-1]
                                    
                        # Update the bracket
                        elif bracket:
                            if np.abs(rupper) < np.abs(rlower):
                                if errors[inewton]*rupper < 0.0:
                                    elower = values[inewton]
                                    rlower = errors[inewton]
                                elif errors[inewton]*rlower < 0.0:
                                    eupper = values[inewton]
                                    rupper = errors[inewton]
                            else:
                                if errors[inewton]*rlower < 0.0:
                                    eupper = values[inewton]
                                    rupper = errors[inewton]
                                elif errors[inewton]*rupper < 0.0:
                                    elower = values[inewton]
                                    rlower = errors[inewton]
                                
                        # Check convergence
                        converged = np.abs(residual) < 0.005/ha2ev or (bracket and np.abs(eupper-elower) < 0.005/ha2ev )
                        
                        # Exit loop
                        if converged:
                            _eout = _ein + 0.0
                            if debug:
                                print(iqp)
                                print_iter(inewton,_ein+efermi[ispin],
                                           _eout+efermi[ispin],
                                           elower+efermi[ispin],
                                           eupper+efermi[ispin],
                                           bracket)
                            break
                        
                        # Decide new Newton step
                        z = -1.0/dresidual
                        step = z*residual
                        
                        if z > 0.3 and z < 1.0:
                            _eout = _ein + step
                        elif bracket and inewton%3 == 0:
                            _eout = elower + 0.6180*(eupper-elower)
                        elif bracket:
                            _eout = eupper - 0.6180*(eupper-elower)
                        elif z > 0.1:
                            _eout = _ein + 0.6180*step
                        else:
                            _eout = _ein + np.sign(residual)*0.005
                                
                        # Print some debug info
                        if debug:
                            print(iqp)
                            print_iter(inewton,
                                       _ein+efermi[ispin],
                                       _eout+efermi[ispin],
                                       elower+efermi[ispin],
                                       eupper+efermi[ispin],
                                       bracket)
                        
                    
                    # Save last energy
                    newevals[ispin][lo[ispin]+iqp] = _eout
                    
                    # Flag current state as converged
                    if converged:
                        fixed[iqp] = True
                    
                    # Estimate remaining error by residual or range of bracket
                    if bracket:
                        esterror[iqp] = min(eupper-elower,np.abs(residual))
                    else:
                        esterror[iqp] = np.abs(residual)
                    
                    
                    if lo[ispin]+iqp < nocc[ispin]:
                        myhi -= 1
                    else:
                        mylo += 1
                        
                    if mylo > myhi:
                        break
                
                # Print output for the states in current cluster
                for jqp in range(llqp,ulqp+1):
                    state = lo[ispin] + jqp + 1
                    print("\t       {:3d}        {:8.3f}       {:8.3f}".format(state,
                             (newevals[ispin][lo[ispin]+jqp]+efermi[ispin])*ha2ev,esterror[jqp]*ha2ev),end="")
                    if fixed[jqp]:
                        print("")
                    else:
                        warning = True
                        print(" ***")
            
            # Print warning label for cases where Newton iteration did not converge
            print("\t      --------------------------------------  ")
            if warning:
                print("\n\t *** Result did not converge\n")

        # Apply scissor shift for evGW and evGW_0 calculations
        if evgw or evgw0:
            scissor(oldevals[0],newevals[0],noqp[0],nvqp[0],nocc[0],lo[0],hi[0],'Alpha')
            if ipol > 1:
                scissor(oldevals[1],newevals[1],noqp[1],nvqp[1],nocc[1],lo[1],hi[1],'Beta')
        
    print("\t Done! ")
    

    
