import os, sys
import numpy as np
#os.system('python3 -m numpy.f2py -c -m mat  generate_matrices.f90 stveco.f90 anxlk.f90 matrxd.f90 matrxo.f90 cd2km.f90 fz.f90 ccrint_new.f90 bessel.f90 ipar.f90 plgndr.f90 w3j.f90')
#os.system('python3 -m numpy.f2py -c -m mat generate_matrices.f90 matrxo.f90 matrxd.f90 anxlk.f90 ccrints.f90 fz.f90 stveco.f90 limits.inc rndoff.inc physcn.inc')
import mat

def conv2coo(filename):
    #x = list(csv.reader(open(filename, "r"), delimiter=delimiter))
    #x = np.array(x).astype(float)
    print('Reading ' + filename)
    x = np.loadtxt(filename)
    dim = int(max(x[:,0]))
    I = x[:,0].astype(int) - 1
    J = x[:,1].astype(int) - 1
    print('Dimensions'+str(dim)+'x'+str(dim))
    E = x[:,2]
    if x.shape[1] > 3: #in case imaginary elements are around
        E = E.astype(complex)
        E += 1.0j*x[:,3] #add the last column if it exists
    return coo_matrix((E,(I,J)),shape=(dim,dim))

simparams_int=np.zeros(9).astype(np.int32)
simparams_double=np.zeros(38)

#gxx(1); gyy(2); gzz(3);axx(4); ayy(5); azz(6);dx(7); dy(8); dz(9);pml(10); pmxy(11); pmzz(12);djf(13); djfprp(14);oss(15); psi(16);ald(17); bed(18); gad(19);alm(20); bem(21); gam(22);
#c20(23); c22(24);c40(25); c42(26); c44(27);t2edi(28);t2ndi(29); t2efi(30);t1edi(31); t1ndi(32);b0(33); a0(34); g0(35);pl(36); pkxy(37);pkzz(38);

simparams_double[0]=2.0087; simparams_double[1]=2.0057; simparams_double[2]=2.0021 #gxx,gyy,gzz
simparams_double[3]=6.0; simparams_double[4]=10.0; simparams_double[5]=36.0; #Axx,Ayy,Azz (Gauss)
simparams_double[6]=5.699; simparams_double[7]=5.699; simparams_double[8]=6;# log10(Rxx) log10(Ryy) log10(Rzz)
simparams_double[15]=10.0; #psi
simparams_double[16]=10.0; simparams_double[17]=15.0; simparams_double[18]=20.0 #ald,bed,gad
simparams_double[19]=25.0; simparams_double[20]=30.0; simparams_double[21]=35.0 #alm,bem,gam
simparams_double[22]=1.0; simparams_double[23]=0.5;#c20,c22
#c40[24]; c42[25]; c44[26];t2edi[27];t2ndi[28]; t2efi[29];t1edi[30]; t1ndi[31];
simparams_double[32]=34050.0 #B0 Gauss
#pl[35]; pkxy[36];pkzz[37];

#in2(1); ipdf(2); ist(3); lemx(4);#lomx(5); kmx(6); mmx(7); ipnmx(8);#ndimo_in(9)
simparams_int[0]=2; simparams_int[1]=0; simparams_int[2]=0; simparams_int[3]=12 #20;
simparams_int[4]=9; simparams_int[5]=4; simparams_int[6]=4; simparams_int[7]=2;
simparams_int[8]=0

zmat_offdiag, zdiag_offdiag, izmat_offdiag, jzmat_offdiag, kzmat_offdiag, zmat_diag, zdiag_diag, \
izmat_diag, jzmat_diag, kzmat_diag, mpid, mpp, stvo, nelreo, nelimo, ndimo, \
nelred, nelimd, ndimd = mat.generate_matrices('xoxo',simparams_double, simparams_int)


from scipy.sparse import csr_matrix, diags, coo_matrix, save_npz
from scipy.sparse.linalg import norm
from scipy.io import mmread, mmwrite
offi=csr_matrix((zmat_offdiag[:nelimo],izmat_offdiag[:nelimo]-1, jzmat_offdiag[:(ndimo+1)]-1))
offr=csr_matrix((zmat_offdiag[:(-nelreo-1):-1],izmat_offdiag[:(-nelreo-1):-1]-1,kzmat_offdiag[:(ndimo+1)]-1))
matx=offr+offr.transpose()+diags(zdiag_offdiag[0,:ndimo])+1.0j*(offi+offi.transpose()+diags(zdiag_offdiag[1,:ndimo]))
matx_actual= conv2coo('matrx_real.mtxx')+1.0j*conv2coo('matrx_imag.mtxx')
#matx_actual=matx_actual.tocsr()

offi=csr_matrix((zmat_diag[:nelimd],izmat_diag[:nelimd]-1, jzmat_diag[:(ndimd+1)]-1))
offr=csr_matrix((zmat_diag[:(-nelred-1):-1],izmat_diag[:(-nelred-1):-1]-1,kzmat_diag[:(ndimd+1)]-1))
print(offi.shape)
print((offi+offi.transpose()+diags(zdiag_diag[1,:ndimd])).shape)
print((offr+offr.transpose()+diags(zdiag_diag[0,:ndimd])).shape)
matz=offr+offr.transpose()+diags(zdiag_diag[0,:ndimd])+1.0j*(offi+offi.transpose()+diags(zdiag_diag[1,:ndimd]))

matz_actual= conv2coo('matrx_real.mtxz')+1.0j*conv2coo('matrx_imag.mtxz')
#np.savetxt('matxz_calculated_real.txt', (offr+offr.transpose()+diags(zdiag_diag[0,:ndimd])).tocoo().toarray())
#print((offr+offr.transpose()+diags(zdiag_diag[0,:ndimd])).tocoo())

print('Deviation between actual and created matrix:',norm(matx-matx_actual))
print('Deviation between actual and created matrix:',norm(matz-matz_actual))
print(np.max(np.abs(matz-matz_actual)))
print((matz-matz_actual).count_nonzero())
print((matz-matz_actual).tocoo())
#print(matx_actual.count_nonzero(), matx.count_nonzero())
#print(matz_actual.count_nonzero(), matz.count_nonzero())
mpp=mpp[:ndimd]
mpid=mpid[:ndimo]
#zmat_offdiag, zdiag_offdiag, izmat_offdiag, jzmat_offdiag, kzmat_offdiag, zmat_diag, zdiag_diag, 
#izmat_diag, jzmat_diag, kzmat_diag, mpid, mpp, stvo, nelreo, nelimo, ndimo, nelred, nelimd, ndimd)
print(nelreo, nelimo, ndimo, nelred, nelimd, ndimd)
#print('Starting vector stveco:')
#print(stvo[:10])
#print('Offdiag izmat, jzmat, kzmat:')
#print('izmat front: ',izmat_offdiag[:10])
#print('izmat back: ',izmat_offdiag[-10:])
#print('jzmat: ',jzmat_offdiag[:10])
#print('kzmat: ',kzmat_offdiag[:10])
#print('zmat:')
#print(zmat_offdiag[:10])
#print(zmat_offdiag[-10:])
#print('zdiag:')
#print(zdiag_offdiag[0,:10])
#print(zdiag_offdiag[1,:10])
#print('Errors with respect to nlspmc_May2018 f77 code: check file test_May18.run for params:')
#print('Off-diag:')
#print(np.max(np.abs(izmat_offdiag)), np.max(np.abs(jzmat_offdiag)), np.max(np.abs(kzmat_offdiag)))
#print(np.max(np.abs(zmat_offdiag)), np.max(np.abs(zdiag_offdiag)))
#print('Diag:')
#print(np.max(np.abs(izmat_diag)), np.max(np.abs(jzmat_diag)), np.max(np.abs(kzmat_diag)))
#print(np.max(np.abs(zmat_diag)), np.max(np.abs(zdiag_diag)))
#print('Diag izmat, jzmat, kzmat:')
#print(izmat_diag[:10])
#print(jzmat_diag[:10])
#print(kzmat_diag[:10])
#print('Diag zmat, zdiag:')
#print(zmat_diag[:10])
#print(zdiag_diag[0,:10])
#print(zdiag_diag[1,:10]) #print(zdiag_diag[:10])
stvx_correct=np.loadtxt('stvec.stvx')
print('Max deviation between actual and generated stvo: ',np.max(np.abs(stvx_correct[:,0]+1.0j * stvx_correct[:,1]-stvo[:ndimo])))
pid_correct=np.loadtxt('pid.txt')
print('Max deviation between actual and generated pid: ',np.max(np.reshape(pid_correct, (-1,1))-np.reshape(mpid, (-1,1))))
pprop_correct=np.loadtxt('pprop.txt')
print('Max deviation between actual and generated pprop: ',np.max(np.reshape(pprop_correct, (-1,1))-np.reshape(mpp, (-1,1))))
#prog2.prog2([2.5,3.5,4.5],[3,4,5],100000000)
