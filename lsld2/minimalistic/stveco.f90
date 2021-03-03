!  VERSION 1.0  (NLSPMC version)   2/5/99
!**********************************************************************
!                       ===================
!                       SUBROUTINE : STVECO
!                       ===================

!     This subroutine calculates the starting vector for the off-diagonal
!     space.  It is intended for use with nonlinear lease-squares
!     applications.

!     On Entry :

!        Parameters are passed through the common block /eprprm/.
!        Basis set is passed through the common block /indexf/.

!     On Exit  :
!        The common block /stvcom/ contains the starting vector for
!        off-diagonal space.

!     Notes:
!        1) The orientational integral is evaluated numerically
!           using the Curtis-Clenshaw-Romberg extrapolation algorithm.
!           For details, see Bruno's thesis.

!     10-APR-93 Sanghyuk Lee

!     Includes :
!        nlsdim.inc
!        eprprm.inc
!        indexf.inc
!        stvcom.inc
!        rndoff.inc

!     Uses:
!        ipar.f
!        ccrint.f

!**********************************************************************
    subroutine stveco(jqe1,pi1,qi1,l1,jk1,k1,jm1,m1,ndimo,ipt,cpot,lptmx,kptmx, stvo)

!    use basis
    include 'limits.inc'
!    include 'simparm.inc'
!    include 'basis.inc'
!    include 'stvcom.inc'
    include 'rndoff.inc'
!    integer,dimension(:),allocatable :: jqe1,pi1,qi1,l1,jk1,k1,jm1,m1 !should already be allocated

    complex*16 stvo(MXDIM)

    logical :: flr,fkr

    integer :: i,iparlr,nup,id,nrow,ipar, &
    iper,jqer,lr,jkr,kr,jmr,mr,ipnr,iqnr,nelv

    double precision :: cnl,stvec,factor,dsq2,vnorm

    double precision :: acc,sml,fz,one,zero,two
    parameter (acc=1.0D-8,sml=1.0D-10)
    parameter (one=1.0D0,zero=0.0D0,two=2.0D0)
    complex*16 czero
    parameter (czero=(0.0d0,0.0d0))

!    common /ifzdat/lr,kr

    external ipar!, fz
 
    integer :: jqe1(MXDIM),pi1(MXDIM),qi1(MXDIM), &
               l1(MXDIM),jk1(MXDIM),k1(MXDIM),jm1(MXDIM), &
               m1(MXDIM)

    integer :: ipt, lptmx, kptmx, ndimo
    double precision :: cpot(5,5)
    dsq2=sqrt(2.0D0)
!----------------------------------------------------------------------
!             initialize counters
!----------------------------------------------------------------------

    vnorm=zero

!----------------------------------------------------------------------
!             *** loop over rows ***
!----------------------------------------------------------------------
!    write(*,*) 'Entering the 1,ndimo loop' 
    do 100 nrow=1,ndimo
    
        stvo(nrow)=czero
    
        jqer=jqe1(nrow)
        lr=l1(nrow)
        jkr=jk1(nrow)
        kr=k1(nrow)
        jmr=jm1(nrow)
        mr=m1(nrow)
        ipnr=pi1(nrow)
        iqnr=qi1(nrow)
    
        if ( (jqer /= 0) .OR. (jkr /= 1) .OR. (jmr /= 1) .OR. &
        (mr /= 0) .OR. (ipnr /= 0) ) go to 100
    
        iparlr=ipar(lr)
        flr=iparlr.eq.1
        cnl=zero
        if(flr) cnl=dsqrt(dble(2*lr+1))
    
        fkr=ipar(kr).eq.1
        stvec=zero
        if(flr .AND. fkr) then
            if(lptmx == 0) then
                if((lr == 0) .AND. (kr == 0)) stvec=one
            else if((kptmx == 0) .AND. (kr /= 0)) then
                stvec=zero
            else
!                    if (nrow.eq.1) write(*,*) 'about to call ccrint_new'    
                call ccrint_new(zero,one,acc,sml,stvec,nup,id,lr,kr,ipt,cpot,lptmx,kptmx)
!                    if (nrow.eq.1) write(*,*) 'called ccrint_new'    
                if(kr /= 0) then
                    factor=one
                    do 500 i=lr-kr+1,lr+kr
                        factor=factor*dble(i)
                    500 END DO
                    factor=one/dsqrt(factor)
                else
                    factor=one/dsq2
                end if
                stvec=factor*stvec*cnl
            end if
        end if
    
    !      **  center of loops  **
    
        stvo(nrow)=dcmplx(stvec,zero)
        vnorm=vnorm+stvec*stvec
    
    100 END DO

!----------------------------------------------------------------------
!     normalize starting vector and zero out imaginary part
!----------------------------------------------------------------------
!   write(*,*) 'Done with the loop, on to counting nelv'
    nelv=0
    vnorm=one/dsqrt(vnorm)
    do 200 i=1,ndimo
        stvo(i)=stvo(i)*vnorm
        if(cdabs(stvo(i)) > rndoff) then
            nelv=nelv+1
        else
            stvo(i)=dcmplx(zero,zero)
        end if
    200 END DO

!----------------------------------------------------------------------
!     zero out remainder of vector
!----------------------------------------------------------------------

!      do 210 i=ndimo+1,mxdim
! 210     stvo(i)=dcmplx(zero,zero)

!----------------------------------------------------------------------
!     return to caller
!----------------------------------------------------------------------


    return
    end subroutine stveco
