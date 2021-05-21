!  VERSION 1.0  (NLSPMC version)   2/5/99
!**********************************************************************

!                       ===================
!                       SUBROUTINE : FBASIS
!                       ===================
!     *** COMBINATION OF FBASO & FBASD ***

!          This routine builds a list of the basis set indices for
!     both diagonal and off-diagonal spaces in common block /indexf/
!     for the truncation parameters set in /eprprm/.
!          If the file, <fileid>.ind exists, it simply reads the basis
!     set.  This basis would be the pruned basis obtained by running
!     program eprbf.  If the file does not exist, it generates full
!     basis set within the given MTS parameters.

!     Includes:
!        nlsdim.inc
!        eprprm.inc
!        basis.inc
!        stdio.inc

!     Uses:
!        ipar.f

!**********************************************************************

    subroutine generate_matrices(ixname,simparams_double_in, simparams_int_in, zmat_offdiag, zdiag_offdiag, &
                             izmat_offdiag, jzmat_offdiag, kzmat_offdiag, zmat_diag, zdiag_diag, &
                             izmat_diag, jzmat_diag, kzmat_diag, mpid, mpp, stvo, nelreo, nelimo, ndimo, &
                             nelred, nelimd, ndimd)
!f2py intent(in) ixname
!f2py intent(in) simparams_double_in
!f2py intent(in) simparams_int_in
!f2py intent(out) zmat_offdiag
!f2py intent(out) zdiag_offdiag
!f2py intent(out) izmat_offdiag
!f2py intent(out) jzmat_offdiag
!f2py intent(out) kzmat_offdiag
!f2py intent(out) zmat_diag
!f2py intent(out) zdiag_diag
!f2py intent(out) izmat_diag
!f2py intent(out) jzmat_diag
!f2py intent(out) kzmat_diag
!f2py intent(out) mpid
!f2py intent(out) mpp
!f2py intent(out) stvo
!f2py intent(out) nelreo
!f2py intent(out) nelimo
!f2py intent(out) ndimo
!f2py intent(out) nelred
!f2py intent(out) nelimd
!f2py intent(out) ndimd


! igen=0 to read from file, else generate full.
!    use basis
    implicit none

    include 'limits.inc'
    include 'rndoff.inc'
    include 'physcn.inc'
!    include 'simparm.inc'
!    include 'basis.inc'
!    include 'stdio.inc'
!    include 'miscel.inc'

    logical :: fexist, setbas
    character*(NAMELG) ixname
    integer :: lu,igen,ierrtmp

    integer :: lr,jkr,kr,krmx,jmr,mr,mrmx,iper,iqer,iqermn,iqermx, &
    ipnr,ipnrmx,ipnrmn,iqnr,iqnrmx,iqnrmn,nrow,i,j,ierr,ilm

    double precision :: sqrt2

    integer :: ipar
    external ipar

    integer :: itmp,inlemx,inlomx,inkmx,inmmx,inpnmx
    double precision :: d2km(2,5,5),tmp,gmax,gmin,hmax

    double precision :: zero,half,dsq23,one,ten
    parameter (zero=0.0D0,half=0.5D0,one=1.0D0,ten=1.0D1)
    parameter (dsq23=0.816496580927726d0)

    logical :: axiala,axialg

    !initialize lstarts_offdiag
    integer :: lstarts_offdiag(MXLVAL+1), lstarts_diag(MXLVAL+1)

    integer :: mjqe1(MXDIM),mpi1(MXDIM),mqi1(MXDIM), &
               ml1(MXDIM),mjk1(MXDIM),mk1(MXDIM),mjm1(MXDIM), &
               mm1(MXDIM)

    integer :: mdjqe1(MXDIM),mdpi1(MXDIM),mdqi1(MXDIM), &
               mdl1(MXDIM),mdjk1(MXDIM),mdk1(MXDIM),mdjm1(MXDIM), &
               mdm1(MXDIM)
    integer :: mpid(MXDIM) !pid(MXDIM),pp(2*MXDIM),
    double precision :: mpp(2*MXDIM)
    complex*16 :: stvo(MXDIM)

    double precision :: zmat_offdiag(MXEL),zdiag_offdiag(2,MXDIM) 
    integer :: izmat_offdiag(MXEL),jzmat_offdiag(MXDIM+1),kzmat_offdiag(MXDIM+1)

    double precision :: zmat_diag(MXEL),zdiag_diag(2,MXDIM)
    integer :: izmat_diag(MXEL),jzmat_diag(MXDIM+1),kzmat_diag(MXDIM+1)


!      double precision, dimension(:,:),allocatable,save:: zdiag
!         allocate(zdiag(2,MXDIM),izmat(MXEL),jzmat(MXDIM+1),&
!          kzmat(MXDIM+1),zmat(MXEL))


    double precision :: simparams_double(38), simparams_double_in(38)
    integer :: simparams_int(8), simparams_int_in(8)

    double precision :: gxx, gyy, gzz, axx, ayy, azz, dx, dy, dz, pml, pmxy, pmzz, djf, djfprp, oss, psi, &
                        ald, bed, gad, alm, bem, gam, c20, c22, c40, c42, &
                        c44, t2edi, t2ndi, t2efi, t1edi, t1ndi, b0, a0, g0, pl,pkxy, pkzz

    integer :: in2, ipdf, ist, igflg, iaflg, irflg, ml, mxy, mzz, lemx, lomx, kmx, mmx, ipnmx, jkmn, jmmn, &
               sptst, ipt, itd, itm, ipsi0, lband, kband, ldelta, kdelta, lptmx, kptmx, &
               nelreo, nelimo, nelred, nelimd, ndimo, ndimd
    double precision :: faa(5), fgm(5), fam(2,5), fgd(2,5), fad(2,5), cpot(5,5), xlk(5,5), zeen, gamman

    gxx=simparams_double_in(1); gyy=simparams_double_in(2); gzz=simparams_double_in(3); 
    axx=simparams_double_in(4); ayy=simparams_double_in(5); azz=simparams_double_in(6); 
    dx=simparams_double_in(7); dy=simparams_double_in(8); dz=simparams_double_in(9); 
    pml=simparams_double_in(10); pmxy=simparams_double_in(11); pmzz=simparams_double_in(12); 
    djf=simparams_double_in(13); djfprp=simparams_double_in(14);
    oss=simparams_double_in(15); psi=simparams_double_in(16); 
    ald=simparams_double_in(17); bed=simparams_double_in(18); gad=simparams_double_in(19);
    alm=simparams_double_in(20); bem=simparams_double_in(21); gam=simparams_double_in(22); 
    c20=simparams_double_in(23); c22=simparams_double_in(24);
    c40=simparams_double_in(25); c42=simparams_double_in(26); c44=simparams_double_in(27); 
    t2edi=simparams_double_in(28);t2ndi=simparams_double_in(29); t2efi=simparams_double_in(30); 
    t1edi=simparams_double_in(31); t1ndi=simparams_double_in(32);
    b0=simparams_double_in(33); a0=simparams_double_in(34); g0=simparams_double_in(35); 
    pl=simparams_double_in(36); pkxy=simparams_double_in(37);
    pkzz=simparams_double_in(38);

    in2=simparams_int_in(1); ipdf=simparams_int_in(2); ist=simparams_int_in(3); lemx=simparams_int_in(4);
    lomx=simparams_int_in(5); kmx=simparams_int_in(6); mmx=simparams_int_in(7); ipnmx=simparams_int_in(8);


    do 29 i=1,(lemx+1)
        lstarts_offdiag(i)=-1
        lstarts_diag(i)=-1
    29 END DO
!######################################################################

          
    ierr=0
    ierrtmp=0

    setbas=.false.
!    if (igen /= 0) go to 100	! generate full
! read from file:
    write(*,*) 'ixname=',ixname
    inquire(file=ixname,exist=fexist)
    if (fexist) then
    
    !----------------------------------------------------------------------
    !     Read basis set information from index file
    !----------------------------------------------------------------------
    
        write(*,*)'opening off-diag space basis index file ', ixname
        open (unit=9,file=ixname,status='old', &
        access='sequential',form='unformatted')
        write(*,*)'opened it'
        read (9) ndimo !ndimoss(nbasis)
        write(*,*) 'ndimo from basis file:',ndimo
        setbas = .true.
    ! fill next area:
        !PG: could do a `wc -l` to know the pruned basis size if fed in as a file
        read (9) (mpi1(i),mqi1(i),ml1(i),mjk1(i),mk1(i),mjm1(i), &
        mm1(i),i=1,ndimo)!pidptr(nbasis),pidptr(nbasis)+ndimoss(nbasis)-1)
        !if (nbasis+1 <= nspectra*ncomps) then   ! ptr to next basis
        !    pidptr(nbasis+1)=pidptr(nbasis)+ndimoss(nbasis)
        !end if
        close (unit=9)
        write(*,*)'closed off-diag space basis index file ', ixname
        !go to 200
    end if

    ierrtmp=1 ! error, gen full basis (assumes have lr, etc.)

!----------------------------------------------------------------------
!     Check the magnetic parameters set in cmds.f
!----------------------------------------------------------------------

!  setbas should be false here since are asking for a basis.
!    ndimo=ndimoss(nbasis)

!    100 continue
!	call pcheck(luttyo,ierr) !start pcheck here!
!fatal error is ierr = -ve and return	
   do 10 j=1,5
        faa(j)=zero
        fgm(j)=zero
        do 9 i=1,2
            fam(i,j)=zero
            fgd(i,j)=zero
            fad(i,j)=zero
        9 END DO
    10 END DO

!----------------------------------------------------------------------
!     Check for zero field
!----------------------------------------------------------------------
    if (b0 < zero) then
        write (*,1002)!ierr=2
        b0=abs(b0)
    elseif (b0 <= rndoff) then
        write (*,2001)!ierr=-1
        return
    endif

!----------------------------------------------------------------------
!     Convert g, A, and R tensors to Cartesian form
!----------------------------------------------------------------------
    !call tocart( gxx, igflg )
    !call tocart( axx, iaflg )
    !call tocart( dx,  irflg )

!  *** NOTE: in NLS version, diffusion rates are specified as
!            a power of 10

!     set criterion for weighting factor to store eigenvector

    tmp=7.99
    do 30 i=1,6
        if ((dx > tmp) .AND. (dy > tmp) .AND. (dz > tmp)) then
            go to 40
        else
            tmp=tmp-1.0d0
        end if
    30 END DO

!  Wtol eliminated.  RC 9/5/96

    40 continue !wtol=1.0D-9

    dx=ten**dx
    dy=ten**dy
    dz=ten**dz

!----------------------------------------------------------------------
!     Check for zero values in g-tensor and zero A tensor
!----------------------------------------------------------------------
    if ( abs(gxx) < rndoff  .OR. &
    abs(gyy) < rndoff  .OR. &
    abs(gzz) < rndoff)   then
        write (*,2002) !ierr=-2
        return
    endif

    if ( (in2 <= 0) .OR. &
    (dabs(axx) < rndoff .AND. &
    dabs(ayy) < rndoff .AND. &
    dabs(azz) < rndoff)     ) then
        in2=0
        axx=zero
        ayy=zero
        azz=zero
        write (*,1001)
        ierr=1
    endif

    gamman=zero

!----------------------------------------------------------------------
!     Calculate spherical components of tensors in magnetic frames
!----------------------------------------------------------------------
!                                        *** Electronic Zeeman ***
    g0=(gxx+gyy+gzz)/3.0D0
    fgm(1)=half*(gxx-gyy)*(b0/g0)
    fgm(3)=dsq23*(gzz-half*(gxx+gyy))*(b0/g0)
    fgm(5)=fgm(1)
    axialg = dabs(fgm(1)).lt.rndoff
!                                        *** Hyperfine ***
    a0=(axx+ayy+azz)/3.0D0
    faa(1)=half*(axx-ayy)

    faa(3)=dsq23*(azz-half*(axx+ayy))
    faa(5)=faa(1)
    axiala = dabs(faa(1)).lt.rndoff
!                                        *** Nuclear Zeeman ***
    zeen=zero

    gmin=dmin1( gxx, gyy, gzz )
    gmax=dmax1( gxx, gyy, gzz )
    hmax=dmax1( dabs(axx), dabs(ayy), dabs(azz) )

!----------------------------------------------------------------------
! Issue warning if high-field approximation has been violated
! (this is a very rough criterion for the high-field approx.!)
!----------------------------------------------------------------------
    if (b0 < 10.0D0*dmax1( hmax, (gmax-gmin)*b0/g0) ) then
        write (*,1003) !ierr=3
    endif
!----------------------------------------------------------------------
!  Set ipt and cpot array according to the potential coefficients given
!----------------------------------------------------------------------
    ipt=0
    lptmx=0
    kptmx=0
    do 240 j=1,5
        do 245 i=1,5
            cpot(i,j)=zero
        245 END DO
    240 END DO

    if (dabs(c20) > rndoff) then
        ipt=ipt+1
        cpot(2,1)=c20
        lptmx=2
    endif
    if (dabs(c22) > rndoff) then
        ipt=ipt+1
        cpot(2,2)=c22
        lptmx=2
        kptmx=2
    endif
    if (dabs(c40) > rndoff) then
        ipt=ipt+1
        cpot(3,1)=c40
        lptmx=4
    endif
    if (dabs(c42) > rndoff) then
        ipt=ipt+1
        cpot(3,2)=c42
        lptmx=4
        kptmx=2
    endif
    if (dabs(c44) > rndoff) then
        ipt=ipt+1
        cpot(3,3)=c44
        lptmx=4
        kptmx=4
    endif

    if (lptmx >= 2) then
        lband=lptmx*2
    else
        lband=2
    end if


!----------------------------------------------------------------------
! Check for consistency of specified motion parameters with given model
!----------------------------------------------------------------------
    if (ipdf > 2) ipdf=2
    if (ipdf < 0) ipdf=0
    if (ist < 0) ist=0

!     Set exponents according to non-Brownian model

    pl=one
    pkxy=one
    pkzz=one
    if (ml == 1)  pl=half
    if (mxy == 1) pkxy=half
    if (mzz == 1) pkzz=half

!     *** Non-Brownian motion: must have at least one nonzero residence
!         time, and no potential specified

! NOTE: in NLS version, R*residence time products are specified as
!       powers of ten

    if (ipdf /= 0) then
        pml = ten**pml
        pmxy = ten**pmxy
        pmzz = ten**pmzz
    
        if ((ml == 0) .AND. (mxy == 0) .AND. (mzz == 0) ) then
            write (*,1004) !ierr=4
            ipdf=0
        end if
    
    !  *** NOTE: in NLS version, djf, djfprp specified as powers of ten
    
        djf=ten**djf
        djfprp=ten**djfprp
    
        if (ipt > 0) then
            write (*,1005) !ierr=5
            ipdf=0
        end if
    end if

!     *** Anisotropic viscosity model: must have potential and
!         no discrete jump motion specified

    if (ipdf == 2) then
        if (ipt == 0) then
            write (*,1006) !ierr=6
            ipdf=0
        end if
    
        if (ist > 0) then
            write (*,1007) !ierr=7
            ipdf=0
        end if
    end if

!     *** Brownian motion model: set residence times to zero

    if (ipdf == 1) then
        if (ml == 0)  pml =zero
        if (mxy == 0) pmxy=zero
        if (mzz == 0) pmzz=zero
    end if

!     *** MOMD calculation: must have potential

!    if (nort > 1) then
!        if (nort > MXORT) then
!            if (lumsg /= 0) write (lumsg,*)'nort too big, reset to ', &
!            MXORT
!            nort=MXORT
!        end if
!        if (ipt < 1) then
!            if (lumsg /= 0) write (lumsg,1015)
!            ierr=15
!        end if
!    end if

!----------------------------------------------------------------------
!     Set director tilt flag
!----------------------------------------------------------------------
    ipsi0=0
!    if (nort > 1) ipsi0=1
    if((abs(psi) > rndoff) .AND. (abs(psi-180.0d0) > rndoff)) ipsi0=1

!----------------------------------------------------------------------
!     Set diffusion tilt flag
!----------------------------------------------------------------------
    bed=abs( bed )
    itd=0
    if ( dabs(ald) > rndoff .OR. dabs(gad) > rndoff &
     .OR. bed > rndoff) itd=1

!----------------------------------------------------------------------
!     Get magnetic tilt angles
!----------------------------------------------------------------------
    bem = dabs( bem )
    if (axialg .AND. dabs(alm) > rndoff) then
        write (*,1013) !ierr=13
        alm=zero
    end if

!     *** Only beta tilt angles allowed if all tensors are axial

    if (axialg .AND. axiala .AND. (dabs(alm) > rndoff .OR. &
    dabs(gam) > rndoff .OR. dabs(gad) > rndoff) ) then
        write (*,1014) !ierr=14
        alm=zero
        gam=zero
        gad=zero
    end if

    itm=0
    if (dabs(alm) > rndoff .OR. bem > rndoff &
     .OR. dabs(gam) > rndoff)     itm=1

!----------------------------------------------------------------------
!     Set A tensor in magnetic (g-tensor) frame
!----------------------------------------------------------------------
    if (itm == 0) then
    !                           *** no tilt: copy tensor directly
        do 252 j=1,5
            fam(1,j)=faa(j)
        252 END DO
    else
    !                          *** transform A tensor into g axis system
    
        call cd2km(d2km,alm,bem,gam)
        do 255 i=1,5
            do 254 j=1,5
                fam(1,i)=fam(1,i)+d2km(1,i,j)*faa(j)
                fam(2,i)=fam(2,i)+d2km(2,i,j)*faa(j)
            254 END DO
        255 END DO
    end if

!----------------------------------------------------------------------
!     Set F_g and F_A tensors in the diffusion frame
!----------------------------------------------------------------------
    if (itd == 0) then
    !                           *** no tilt: copy tensors directly
        do 260 j=1,5
            fgd(1,j)=fgm(j)
            do 259 i=1,2
                fad(i,j)=fam(i,j)
            259 END DO
        260 END DO
    
    else
    !                    *** Transform A and g tensors into the diffusion frame
    
        call cd2km(d2km,ald,bed,gad)
        do 265 i=1,5
            do 264 j=1,5
                fgd(1,i)=fgd(1,i)+d2km(1,i,j)*fgm(j)
                fgd(2,i)=fgd(2,i)+d2km(2,i,j)*fgm(j)
                fad(1,i)=fad(1,i)+d2km(1,i,j)*fam(1,j) &
                -d2km(2,i,j)*fam(2,j)
                fad(2,i)=fad(2,i)+d2km(1,i,j)*fam(2,j) &
                +d2km(2,i,j)*fam(1,j)
            264 END DO
        265 END DO
    end if

!----------------------------------------------------------------------
!     Relaxation parameters & Heisenberg exchange parameter
!----------------------------------------------------------------------

! NOTE: in NLS version, these rates are given in powers of ten

    if (t2edi > rndoff) t2edi=ten**t2edi
    if (t2ndi > rndoff) t2ndi=ten**t2ndi
    if (t2efi > rndoff) t2efi=ten**t2efi
    if (t1edi > rndoff) t1edi=ten**t1edi
    if (t1ndi > rndoff) t1ndi=ten**t1ndi
    if (oss > rndoff) oss=ten**oss

!----------------------------------------------------------------------
!     Check basis set parameters
!----------------------------------------------------------------------
!                             ** basis already set **
    if (setbas) go to 200

    inlemx = lemx
    inlomx = lomx
    inkmx  = kmx
    inmmx  = mmx
    inpnmx = ipnmx

    if((lemx > mxlval)) lemx=mxlval
    if(ipar(lemx) /= 1) lemx=lemx-1
    if(ipar(lomx) /= -1) lomx=lomx-1
    if(lomx > lemx) lomx=lemx-1
    if(kmx > lemx) kmx=lemx
    if(mmx > lemx) mmx=lemx
    if(ipar(kmx) /= 1) kmx=kmx-1
    if(ipnmx > in2) ipnmx=in2
    if((ipsi0 == 0) .AND. (mmx > ipnmx+1)) mmx=ipnmx

    if(lemx < 0)  lemx=0
    if(lomx < 0)  lomx=0
    if(kmx < 0)   kmx=0
    if(mmx < 0)   mmx=0
    if(ipnmx < 0) ipnmx=0

    if (inlemx /= lemx .OR. &
    inlomx /= lomx .OR. &
    inkmx  /= kmx  .OR. &
    inmmx  /= mmx  .OR. &
    inpnmx /= ipnmx ) then
    
        write (*,1009) lemx,lomx,kmx,mmx,ipnmx !ierr=9
    endif

!----------------------------------------------------------------------
!  Determine basis set using rules in M,I,I,M & Freed
!  (1) jm=1 (use only symmetric M combinations) for no nuclear Zeeman
!  (2) jkmn=1 (use only symmetric K combinations) if the magnetic
!          tensors in the diffusion frame are real-valued. This is the
!          case if alpha_m, gamma_m (magnetic tilt), and gamma_d
!          (diffusion tilt) are all zero.
!  (3) only even K values if there is no magnetic and diffusion tilt.
!  (4) only even L values no K values (kmx=0) in case of axial magnetic
!          tensors, axial potential, and no magnetic/diffusion tilt.
!----------------------------------------------------------------------
    jmmn=1
    if (abs(zeen) > rndoff) jmmn=-1

    jkmn=1
    do 270 i=1,5
        if (    dabs(fgd(2,i)) >= rndoff &
         .OR. dabs(fad(2,i)) >= rndoff ) jkmn=-1
    270 END DO

    if (itm == 0 .AND. itd == 0) then
        kdelta=2
    else
        kdelta=1
    end if

    if (axiala .AND. axialg .AND. kdelta == 2 .AND. kptmx == 0) &
    then
        ldelta=2
        lomx=0
        kmx=0
    else
        ldelta=1
    end if

!    if (ierr < 0) then
!        ierr=3
!        return
!    end if
500   continue
    nrow=0 !pidptr(nbasis)-1


! OFF DIAGONAL BASIS: 

!----------------------------------------------------------------------
!     *** loop over lr ***
!----------------------------------------------------------------------

    do 110 lr=0,lemx,ldelta
        lstarts_offdiag(lr+1)=nrow+1
        if((ipar(lr) /= 1) .AND. (lr > lomx)) go to 110
    
    !----------------------------------------------------------------------
    !     *** loop over jkr ***
    !----------------------------------------------------------------------
    
        do 120 jkr=jkmn,1,2
        
        !----------------------------------------------------------------------
        !     *** loop over kr ***
        !----------------------------------------------------------------------
        
            krmx=min(kmx,lr)
            do 130 kr=0,krmx,kdelta
                if((kr == 0) .AND. (ipar(lr) /= jkr)) go to 130
            
            !----------------------------------------------------------------------
            !     *** loop over jmr ***
            !----------------------------------------------------------------------
            
                do 140 jmr=jmmn,1,2
                
                !----------------------------------------------------------------------
                !     *** loop over mr ***
                !----------------------------------------------------------------------
                
                    mrmx=min(mmx,lr)
                    do 150 mr=0,mrmx
                    
                    !----------------------------------------------------------------------
                    !     *** loop over ipnr ***
                    !----------------------------------------------------------------------
                    
                        ipnrmx=min(in2,ipnmx)
                        if (mr == 0) then
                            ipnrmn=0
                        else
                            ipnrmn=-ipnrmx
                        end if
                    
                        do 160 ipnr=ipnrmn,ipnrmx
                            if((mr == 0) .AND. (ipnr == 0) .AND. (ipar(lr) /= jmr)) &
                            go to 160
                            if((ipsi0 == 0) .AND. (ipnr /= mr)) go to 160
                        
                        !----------------------------------------------------------------------
                        !     *** loop over iqnr ***
                        !----------------------------------------------------------------------
                        
                            iqnrmx=in2-iabs(ipnr)
                            iqnrmn=-iqnrmx
                            do 170 iqnr=iqnrmn,iqnrmx,2
                            
                                nrow=nrow+1
                                mjqe1(nrow)=0
                                ml1(nrow)=lr
                                mjk1(nrow)=jkr
                                mk1(nrow)=kr
                                mjm1(nrow)=jmr
                                mm1(nrow)=mr
                                mpi1(nrow)=ipnr
                                mqi1(nrow)=iqnr
                            
                            !----------------------------------------------------------------------
                            !     end loop over rows
                            !----------------------------------------------------------------------
                            
                            170 END DO
                        160 END DO
                    150 END DO
                140 END DO
            130 END DO
        120 END DO
    110 END DO
    ndimo=nrow ! size of this one

    !write to an unformatted file
    open (unit=9,file='ind_offdiag.indx',status='unknown',access='sequential', form='unformatted') 
    !PG writing indices to file; activate when needed, not writing ndimo
    write(9) ndimo
    write(9) (mpi1(i),mqi1(i),ml1(i),mjk1(i),mk1(i),mjm1(i),mm1(i),i=1,ndimo)
    close(unit=9)

!    if (nbasis+1 <= nspectra*ncomps) then   ! ptr to next basis
!        pidptr(nbasis+1)=nrow+1		! location of next basis
!    end if


    200 continue

!      ndimo=nrow
!corrected parameters when needed, now put them back into the simparams arrays so that they can be passed to other subroutines 
    simparams_double(1)=gxx; simparams_double(2)=gyy; simparams_double(3)=gzz;
    simparams_double(4)=axx; simparams_double(5)=ayy; simparams_double(6)=azz;
    simparams_double(7)=dx; simparams_double(8)=dy; simparams_double(9)=dz;
    simparams_double(10)=pml; simparams_double(11)=pmxy; simparams_double(12)=pmzz;
    simparams_double(13)=djf; simparams_double(14)=djfprp;
    simparams_double(15)=oss; simparams_double(16)=psi;
    simparams_double(17)=ald; simparams_double(18)=bed; simparams_double(19)=gad;
    simparams_double(20)=alm; simparams_double(21)=bem; simparams_double(22)=gam;
    simparams_double(23)=c20; simparams_double(24)=c22;
    simparams_double(25)=c40; simparams_double(26)=c42; simparams_double(27)=c44;
    simparams_double(28)=t2edi;simparams_double(29)=t2ndi; simparams_double(30)=t2efi;
    simparams_double(31)=t1edi; simparams_double(32)=t1ndi;
    simparams_double(33)=b0; simparams_double(34)=a0; simparams_double(35)=g0;
    simparams_double(36)=pl; simparams_double(37)=pkxy; simparams_double(38)=pkzz

    simparams_int(1)=in2; simparams_int(2)=ipdf; simparams_int(3)=ist; simparams_int(4)=lemx;
    simparams_int(5)=lomx; simparams_int(6)=kmx; simparams_int(7)=mmx; simparams_int(8)=ipnmx;

!**********************************************************************
!     Basis index for diagonal space (FBASD)
!**********************************************************************

!    one=1.0D0
    sqrt2=dsqrt(2.0D0)

!----------------------------------------------------------------------
!       convert pruned off-diagonal basis set into diagonal one.
!----------------------------------------------------------------------

!    open (unit=9,file='ind_diag.indx',status='unknown', &
!          access='sequential') !PG writing indices to file; activate when needed
    j=0!dpidptr(nbasis)-1
    do 210 i=1,ndimo !pidptr(nbasis),pidptr(nbasis)+ndimoss(nbasis)-1
        j=j+1
        mdl1(j)=ml1(i)
        if(j.eq.1 .or. mdl1(j).gt.mdl1(j-1)) lstarts_diag(mdl1(j)+1)=j
        mdjk1(j)=mjk1(i)
        mdk1(j)=mk1(i)
        mdjm1(j)=0
        mdm1(j)=mm1(i)
        mdjqe1(j)=mjm1(i)
        mdpi1(j)=mpi1(i)
        mdqi1(j)=mqi1(i)
!        write(9,*) j,mdl1(j),mdjk1(j),mdk1(j), &
!                   mdjm1(j),mdm1(j),mdjqe1(j),mdpi1(j),mdqi1(j)
        if ((mpi1(i) == 0) .AND. (mm1(i) == 0)) then
            mpid(i)=1
            mpp(j)=sqrt2
        else
            mpp(j)=one
        
            j=j+1
            mdl1(j)=ml1(i)
            mdjk1(j)=mjk1(i)
            mdk1(j)=mk1(i)
            mdjm1(j)=0
            mdm1(j)=-mm1(i)
            mdjqe1(j)=mjm1(i)
            mdpi1(j)=-mpi1(i)
            mdqi1(j)=mqi1(i)
            mpid(i)=2
            ilm=ipar(ml1(i)+mm1(i))
            if (ilm == mjm1(i)) then
                mpp(j)=one
            else
                mpp(j)=-one
            end if
!            write(9,*) j,mdl1(j),mdjk1(j),mdk1(j), &
!                       mdjm1(j),mdm1(j),mdjqe1(j),mdpi1(j),mdqi1(j)
        end if
    210 END DO
    do 211 i=ndimo+1,j !pidptr(nbasis)+ndimoss(nbasis),j
        mpid(i)=-1 ! test for problems
    211 END DO
!    if (nbasis+1 <= nspectra*ncomps) then   ! ptr to next basis
!        dpidptr(nbasis+1)=j+1		! location of next basis
!    end if
    !ndimdss(nbasis)=j-dpidptr(nbasis)+1	! length of dp
    ndimd=j
!    close(unit=9)

    if (ierrtmp == 1) ierr=1
!    if ((pidptr(nbasis) > mmxdim) .OR. (dpidptr(nbasis) &
!     > 2*mmxdim)) ierr=2
   
!    write(*,*) 'ldelta=',ldelta
    do 6363 i=1,lemx
        if(lstarts_diag(i).eq.-1) lstarts_diag(i)=lstarts_diag(i+1)
        if(lstarts_offdiag(i).eq.-1) lstarts_offdiag(i)=lstarts_offdiag(i+1)
    6363 END DO
!    lstarts_offdiag(lemx+1)=ndimo
!    lstarts_diag(lemx+1)=ndimd !ndimdss(nbasis)

!    open (unit=9,file='lstarts_diag.indx',status='unknown', &
!          access='sequential') !PG writing indices to file; activate when needed
!    do 6364 i=1,(lemx+1)
!        write(9,*) i-1, lstarts_diag(i)
!    6364 END DO
!    write(9,*) lemx, lstarts_diag(i)
!    close(unit=9)
!    write(*,*) 'Ordering potential parameters:'
 !   write(*,*) ipt, lptmx, kptmx
    call matrxo(simparams_double, simparams_int, fad, fgd, lstarts_offdiag, &
                      mjqe1,mpi1,mqi1,ml1,mjk1,mk1,mjm1,mm1,ndimo,&
                      zmat_offdiag,zdiag_offdiag,izmat_offdiag,&
                      jzmat_offdiag,kzmat_offdiag,nelimo,nelreo,ipt,cpot,lptmx,kptmx, setbas)
    call matrxd(simparams_double, simparams_int, fad, fgd, lstarts_diag, &
                      mdjqe1,mdpi1,mdqi1,mdl1,mdjk1,mdk1,mdjm1,mdm1, ndimd, &
                       zmat_diag, zdiag_diag, izmat_diag, jzmat_diag, &
                       kzmat_diag, nelimd, nelred, ipt,cpot,lptmx,kptmx, setbas)
!    write(*,*) 'cpot(2,1)=',cpot(2,1)
    call stveco(mjqe1,mpi1,mqi1,ml1,mjk1,mk1,mjm1,mm1,ndimo,ipt,cpot,lptmx,kptmx,stvo)
    
    1001 format(' Nuclear spin in2=0 assumed')
    1002 format(' Positive B0 value assumed')
    1003 format(' Warning: high-field approximation may not apply for', &
    ' given parameters')
    1004 format(' No non-Brownian motion specified for l, xy, zz:', &
    ' ipdf=0 assumed')
    1005 format(' Nonzero potential with jump/free model:', &
    ' ipdf=0 assumed')
    1006 format(' Zero potential with anisotropic viscosity:', &
    ' ipdf=0 assumed')
    1007 format(' Discrete jumps with anisotropic viscosity:', &
    ' ipdf=0 assumed')
    1009 format(' Basis set parameters adjusted to ',4(i3,','), i3)
    1013 format(' Axial g-tensor: alm=0 assumed')
    1014 format(' Axial g, hf-tensors: alm=gam=gad=0 assumed')
    2001 format(' Fatal parameter error: zero B0')
    2002 format(' Fatal parameter error: zero values in g-tensor')
    
end subroutine generate_matrices

