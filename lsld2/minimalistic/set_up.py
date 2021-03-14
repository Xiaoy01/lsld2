# The Fortran90 wrapped module just generated, mat.generate_matrices is a F90
# subroutine that can be called in Python
import mat
import os
import numpy as np
from scipy.sparse import csr_matrix, diags, coo_matrix
from scipy.sparse.linalg import norm
# Install the "mat" package from the Fortran90 files listed below using the
# Numpy f2py wrapper
# See "quick and smart way" at https://numpy.org/doc/stable/f2py/f2py.getting-started.html#the-quick-and-smart-way # noqa
# Ignore the warnings generated during numpy.f2py; have a lot of unused
# variables; don't know why the "uninitialized-maybe" warnings come
os.system('python3 -m numpy.f2py -c -m mat  generate_matrices.f90 stveco.f90'
          ' anxlk.f90 matrxd.f90 matrxo.f90 cd2km.f90 fz.f90 ccrint_new.f90'
          ' bessel.f90 ipar.f90 plgndr.f90 w3j.f90')


def conv2coo(filename):
    '''
    misc function that converts <filename> to a scipy.sparse.COO matrix;
    assuming the file comes from a language like Fortran/Matlab that begins
    indexing from 1 to convert to Python we need to subtract indices by 1, b/c
    Python starts arrays from 0
    Input: filename string
    Output: COO matrix
    '''
    print('Reading ' + filename)
    x = np.loadtxt(filename)
    dim = int(max(x[:, 0]))
    I = x[:, 0].astype(int) - 1
    J = x[:, 1].astype(int) - 1
    print('Dimensions ' + str(dim) + 'x' + str(dim))
    E = x[:, 2]
    return coo_matrix((E, (I, J)), shape=(dim, dim))


def generate_from_params(offdiag_basis_file, simparams_double, simparams_int):
    '''
    Input:
    1. offdiag_basis_file: input basis set (a nonsensical, non-existent file
    like 'xoxo' means the F90 subroutine will assume that you are generating
    matrices, etc. from scratch)
    2. simparams_double: simulation parameters (look Budil et al 1996 Table 1
    to find the meanings of each parameter gxx(1); gyy(2); gzz(3); axx(4);
    ayy(5); azz(6); dx(7); dy(8); dz(9); pml(10); pmxy(11); pmzz(12); djf(13);
    djfprp(14); oss(15); psi(16); ald(17); bed(18); gad(19); alm(20); bem(21);
    gam(22); c20(23); c22(24); c40(25); c42(26); c44(27); t2edi(28); t2ndi(29);
    t2efi(30); t1edi(31); t1ndi(32); b0(33); a0(34); g0(35); pl(36); pkxy(37);
    pkzz(38);
    3. simparams_int: more simulation parameters that are integers, not double
    precision numbers in2(1); ipdf(2); ist(3); lemx(4);lomx(5); kmx(6); mmx(7);
    ipnmx(8);ndimo_in(9)

    Output:
    matx: off-diag space matrix in scipy.csr format
    matz: diag space matrix in scipy.csr format
    pp: pulse propagator in scipy.csr format
    stvx: off-diag space starting vector

    Notes:
    1. <ndimo_in> is a new parameter (not mentioned in Budil 1996) that
    indicates the size of the input basis set in the <offdiag_basis_file> file;
    this parameter has no meaning if <offdiag_basis_file> has a nonsensical
    name or doesn't exist in the directory
    Returns:
    2. To people new to SLE, off-diagonal and diagonal spaces are DIFFERENT
    from off-diagonal and diagonal parts of a matrix. At a deeper level there
    is a relation, but you need to know density matrices in quantum mechanics.
    '''
    # 'xoxo' is a dummy file
    zmat_offdiag, zdiag_offdiag, izmat_offdiag, jzmat_offdiag, kzmat_offdiag, \
        zmat_diag, zdiag_diag, izmat_diag, jzmat_diag, kzmat_diag, mpid, mpp, \
        stvx, nelreo, nelimo, ndimo, nelred, nelimd, ndimd = \
        mat.generate_matrices('xoxo', simparams_double, simparams_int)
    # off-diag space starting vector <stvx>
    # such truncations viz. [:ndimo] are needed, because F90 f2py doesn't allow
    # allocatable arrays it seems, so we need to define a much larger array
    # https://stackoverflow.com/questions/34579769/f2py-error-with-allocatable-arrays/34708146
    stvx = stvx[:ndimo]
    # off-diag space SLE matrix <matx>
    offi = csr_matrix((zmat_offdiag[:nelimo],
                       izmat_offdiag[:nelimo] - 1,
                       jzmat_offdiag[:(ndimo + 1)] - 1))
    offr = csr_matrix((zmat_offdiag[:(-nelreo - 1):-1],
                       izmat_offdiag[:(-nelreo - 1):-1] - 1,
                       kzmat_offdiag[:(ndimo + 1)] - 1))
    matx = offr + offr.transpose() + diags(zdiag_offdiag[0, :ndimo]) + 1.0j * \
        (offi + offi.transpose() + diags(zdiag_offdiag[1, :ndimo]))
    # diag space SLE matrix <matz>
    offi = csr_matrix((zmat_diag[:nelimd],
                       izmat_diag[:nelimd] - 1,
                       jzmat_diag[:(ndimd + 1)] - 1))
    offr = csr_matrix((zmat_diag[:(-nelred - 1):-1],
                       izmat_diag[:(-nelred - 1):-1] - 1,
                       kzmat_diag[:(ndimd + 1)] - 1))
    matz = offr + offr.transpose() + diags(zdiag_diag[0, :ndimd]) + 1.0j * \
        (offi + offi.transpose() + diags(zdiag_diag[1, :ndimd]))
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
                    shape=(ndimd, ndimo)).tocsr()
    return matx, matz, pp, stvx


def test_matrices():
    '''
    checks if the generated matrices match the expected matrices; expected
    matrices come from nlspmc_May2018EverythingTimeDomain on the ELDOR machine;
    to see the parameters that were used in the F77
    nlspmc_May2018EverythingTimeDomain code, check "test_file.run"
    '''
    # Initialize parameters, then fill in
    testparams_int = np.zeros(9).astype(np.int32)
    testparams_double = np.zeros(38)  # initialize parameters, then fill in
    testparams_double[0] = 2.0087
    testparams_double[1] = 2.0057
    testparams_double[2] = 2.0021  # gxx,gyy,gzz
    testparams_double[3] = 6.0
    testparams_double[4] = 10.0
    testparams_double[5] = 36.0  # Axx,Ayy,Azz (Gauss)
    testparams_double[6] = 5.699
    testparams_double[7] = 5.699
    testparams_double[8] = 6  # log10(Rxx) log10(Ryy) log10(Rzz)
    testparams_double[15] = 10.0  # psi
    testparams_double[16] = 10.0
    testparams_double[17] = 15.0
    testparams_double[18] = 20.0  # ald,bed,gad
    testparams_double[19] = 25.0
    testparams_double[20] = 30.0
    testparams_double[21] = 35.0  # alm,bem,gam
    testparams_double[22] = 1.0
    testparams_double[23] = 0.5  # c20,c22
    testparams_double[32] = 34050.0  # B0 Gauss
    # c40[24]; c42[25]; c44[26];#t2edi[27];t2ndi[28]; t2efi[29];#t1edi[30]; t1ndi[31];#pl[35]; pkxy[36];#pkzz[37]; # noqa
    testparams_int[0] = 2
    testparams_int[1] = 0
    testparams_int[2] = 0
    testparams_int[3] = 12  # 20
    testparams_int[4] = 9
    testparams_int[5] = 4
    testparams_int[6] = 4
    testparams_int[7] = 2
    testparams_int[8] = 0
    # load the expected test files
    # matrx_real, matrx_imag have the real and imag parts of the reference
    # off-diagonal/diagonal SLE matrix
    # .mtxx means off-diagonal space, .mtxz means diagonal space
    matx_correct = conv2coo('matrx_real.mtxx') + 1.0j * \
        conv2coo('matrx_imag.mtxx').tocsr()
    matz_correct = conv2coo('matrx_real.mtxz') + 1.0j * \
        conv2coo('matrx_imag.mtxz').tocsr()
    stvx_correct = np.loadtxt('stvec.stvx')
    # generate the matrices
    matx, matz, pp, stvx = generate_from_params('xoxo',
                                                testparams_double,
                                                testparams_int)
    # compare the 2: ideal deviation should be 0
    print('Max deviation between correct and generated off-diag space matrix:',
          norm(matx - matx_correct))
    print('Max deviation between correct and generated diag space matrix:',
          norm(matz - matz_correct))
    print('Max deviation between correct and generated stvx: ',
          np.max(np.abs(stvx_correct[:, 0] + 1.0j * stvx_correct[:, 1]
                        - stvx)))
    pid_correct = np.loadtxt('pid.txt')
    pprop_correct = np.loadtxt('pprop.txt')
    indx = []
    for k in range(matx_correct.shape[0]):
        if pid_correct[k] == 1:
            indx.append(k)
        elif pid_correct[k] == 2:
            indx.append(k)
            indx.append(k)
    pp_correct = coo_matrix((pprop_correct, (np.arange(matz_correct.shape[0]),
                                             np.array(indx))), shape=(
        matz_correct.shape[0], matx_correct.shape[0])).tocsr()
    print('Max deviation between correct and generated pulse propagator: ',
          norm(pp - pp_correct))


test_matrices()
