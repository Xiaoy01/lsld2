import mat  # the Fortran90 wrapped module just generated, mat.generate_matrices is a F90 subroutine that can be called in Python
import os
import numpy as np
from scipy.sparse import csr_matrix, spdiags, coo_matrix

# Hyperparameters
MIN_LEMX = 8  # 16
MXDIM = 2000
MXEL = 300 * MXDIM
KRYLOV_THRESH = 2000  # 0
MXPT = 4096

# modify limits.inc; going for a small MXDIM as X band non-ultra-slow motions
f = open('limits.inc', 'w')
f.write('''
    integer :: MXDIM,MXEL,MXLVAL,NAMELG,MMAX,MXPT
    parameter (MXDIM={},MXEL={},MXLVAL=200,NAMELG=40,MMAX=100,MXPT={})
'''.format(MXDIM, MXEL, MXPT))
f.close()

# install the "mat" package from the Fortran90 files listed below using the Numpy f2py wrapper; this is Windows version!!!!
# see "quick and smart way" at https://numpy.org/doc/stable/f2py/f2py.getting-started.html#the-quick-and-smart-way
# ignore the warnings generated during numpy.f2py; have a lot of unused variables; don't know why the "uninitialized-maybe" warnings come
print('No need to compile the Fortran code again and again, go to setup.py and comment the f2py line to avoid wasting time every compilation')
# compile the mat module
if not os.path.exists("mat.cpython-36m-darwin.so"):
    os.system('f2py -c -m mat  generate_matrices.f90 stveco.f90 anxlk.f90 matrxd.f90 matrxo.f90 cd2km.f90 fz.f90 ccrint_new.f90 bessel.f90 ipar.f90 plgndr.f90 w3j.f90')
if not os.path.exists("gconvl.cpython-36m-darwin.so"):
    os.system('f2py -c -m gconvl  gconvl.f')  # compile the gconvl Gaussian convolution module

# definitions of parameters as per their positions in the arrays fed to Fortran
params_double_def = {'gxx': 0, 'gyy': 1, 'gzz': 2, 'axx': 3, 'ayy': 4, 'azz': 5, 'dx': 6, 'dy': 7, 'dz': 8,
                     'pml': 9, 'pmxy': 10, 'pmzz': 11, 'djf': 12, 'djfprp': 13, 'oss': 14, 'psi': 15,
                     'ald': 16, 'bed': 17, 'gad': 18, 'alm': 19, 'bem': 20, 'gam': 21,
                     'c20': 22, 'c22': 23, 'c40': 24, 'c42': 25, 'c44': 26,
                     't2edi': 27, 't2ndi': 28, 't2efi': 29, 't1edi': 30, 't1ndi': 31,
                     'b0': 32, 'a0': 33, 'g0': 34, 'pl': 35, 'pkxy': 36, 'pkzz': 37}
params_int_def = {'in2': 0, 'ipdf': 1, 'ist': 2,
                  'lemx': 3, 'lomx': 4, 'kmx': 5, 'mmx': 6, 'ipnmx': 7}

# suggest a decent basis size as per Budil's equation; for fast motions best is to use lemx=8 at X band


def budil_basis_size(rbar, B0):
    lemx = round((3.07+0.0068*0.0028*B0)*(8.40+0.0075*0.0028*B0-rbar+0.3010)**2)
    lemx = MIN_LEMX if lemx < MIN_LEMX else lemx
    lemx += lemx % 2
    lomx = round(0.7*lemx)
    lomx += lomx % 2+1
    kmx = round(lemx*0.5)
    kmx += kmx % 2
    mmx = kmx
    return lemx, lomx, kmx, mmx


def conv2coo(filename):
    '''
    misc function that converts <filename> to a scipy.sparse.COO matrix;
    assuming the file comes from a language like Fortran/Matlab that begins indexing from 1
    to convert to Python we need to subtract indices by 1, b/c Python starts arrays from 0

    Input: filename string
    Output: COO matrix
    '''
    print('Reading ' + filename)
    x = np.loadtxt(filename)
    dim = int(max(x[:, 0]))
    I = x[:, 0].astype(int) - 1
    J = x[:, 1].astype(int) - 1
    print('Dimensions '+str(dim)+'x'+str(dim))
    E = x[:, 2]
    return coo_matrix((E, (I, J)), shape=(dim, dim))


def generate_from_params(offdiag_basis_file, simparams_double, simparams_int):
    '''
    Input:
    1. offdiag_basis_file: input basis set (a nonsensical, non-existent file like 'xoxo' means the F90 subroutine
    will assume that you are generating matrices, etc. from scratch)
    2. simparams_double: simulation parameters (look Budil et al 1996 Table 1 to find the meanings of each parameter
    gxx(1); gyy(2); gzz(3);axx(4); ayy(5); azz(6);dx(7); dy(8); dz(9);pml(10); pmxy(11); pmzz(12);djf(13); djfprp(14);oss(15); psi(16);ald(17); bed(18); gad(19);alm(20); bem(21); gam(22);
    c20(23); c22(24);c40(25); c42(26); c44(27);t2edi(28);t2ndi(29); t2efi(30);t1edi(31); t1ndi(32);b0(33); a0(34); g0(35);pl(36); pkxy(37);pkzz(38);
    3. simparams_int: more simulation parameters that are integers, not double precision numbers
    in2(1); ipdf(2); ist(3); lemx(4);lomx(5); kmx(6); mmx(7); ipnmx(8)

    Output:
    matx: off-diag space matrix in scipy.csr format
    matz: diag space matrix in scipy.csr format
    pp: pulse propagator in scipy.csr format
    stvx: off-diag space starting vector

    Notes:
    1. <ndimo_in> has been removed, PG 2/17/21
    2. To people new to SLE, off-diagonal and diagonal spaces are DIFFERENT from off-diagonal and diagonal parts of a matrix.
    At a deeper level there is a relation, but you need to know density matrices in quantum mechanics.
    '''
    # print(simparams_int)
    zmat_offdiag, zdiag_offdiag, izmat_offdiag, jzmat_offdiag, kzmat_offdiag, zmat_diag, zdiag_diag, \
        izmat_diag, jzmat_diag, kzmat_diag, mpid, mpp, stvx, nelreo, nelimo, ndimo, \
        nelred, nelimd, ndimd = mat.generate_matrices(
            offdiag_basis_file, simparams_double, simparams_int)  # 'xoxo' is a dummy file
    #print(nelreo, nelimo, ndimo, nelred, nelimd, ndimd)
    # off-diag space starting vector <stvx>
    # such truncations viz. [:ndimo] are needed, because F90 f2py doesn't allow allocatable arrays it seems, so we need to define a much larger array
    # https://stackoverflow.com/questions/34579769/f2py-error-with-allocatable-arrays/34708146
    stvx = stvx[:ndimo]
    # off-diag space SLE matrix <matx>
    offi = csr_matrix((zmat_offdiag[:nelimo], izmat_offdiag[:nelimo]-1,
                       jzmat_offdiag[:(ndimo+1)]-1), shape=(ndimo, ndimo))
    if nelreo > 0:
        offr = csr_matrix((zmat_offdiag[:(-nelreo-1):-1], izmat_offdiag[:(-nelreo-1)                                                                        :-1]-1, kzmat_offdiag[:(ndimo+1)]-1), shape=(ndimo, ndimo))
    else:
        offr = csr_matrix((ndimo, ndimo))
    matx = offr+offr.transpose()+spdiags(zdiag_offdiag[0, :ndimo], [0], ndimo, ndimo)+1.0j*(
        offi+offi.transpose()+spdiags(zdiag_offdiag[1, :ndimo], [0], ndimo, ndimo))
    # diag space SLE matrix <matz>
    offi = csr_matrix((zmat_diag[:nelimd], izmat_diag[:nelimd]-1,
                       jzmat_diag[:(ndimd+1)]-1), shape=(ndimd, ndimd))
    if nelred > 0:
        offr = csr_matrix((zmat_diag[:(-nelred-1):-1], izmat_diag[:(-nelred-1):-1]-1, kzmat_diag[:(ndimd+1)]-1), shape=(ndimd, ndimd))
    else:
        offr = csr_matrix((ndimd, ndimd))
    #print(offr.shape, spdiags(zdiag_diag[0,:ndimd],[0],ndimd,ndimd).shape, spdiags(zdiag_diag[1,:ndimd],[0],ndimd,ndimd).shape, offi.shape)
    matz = offr+offr.transpose()+spdiags(zdiag_diag[0, :ndimd], [0], ndimd, ndimd)+1.0j*(
        offi+offi.transpose()+spdiags(zdiag_diag[1, :ndimd], [0], ndimd, ndimd))
    # pulse propagator <pp>
    mpp = mpp[:ndimd]
    mpid = mpid[:ndimo]
    indx = []
    for k in range(ndimo):
        if mpid[k] == 1:
            indx.append(k)
        elif mpid[k] == 2:
            indx.append(k)
            indx.append(k)
    pp = coo_matrix((mpp, (np.arange(ndimd), np.array(indx))),
                    shape=(ndimd, ndimo)).tocsr()/np.sqrt(2)
    return matx, matz, pp, stvx  # , [zmat_offdiag, zdiag_offdiag, izmat_offdiag, jzmat_offdiag, kzmat_offdiag],\
    # [zmat_diag, zdiag_diag, izmat_diag, jzmat_diag, kzmat_diag]
