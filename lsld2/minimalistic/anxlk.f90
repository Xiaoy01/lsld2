!  VERSION 1.0  (NLSPMC version)   2/5/99
!**********************************************************************
!                         ====================
!                           subroutine ANXLK
!                         ====================

!       This subroutine calculates the xlk coefficients for
!       the potential-dependent part of the diffusion operator
!       for use in the SLE matrix calculation (MATRL, MATRF).
!       Array xlk (in /eprdat/) should be dimensioned to at least
!       (5,5), so that it can accommodate even L,K values from 0
!       to 8 (twice the highest possible L of a potential coefficient)

!       Notes:
!          The summation over the X(L,K)'s proceeds from 0 to 2*lptmx
!          X(L,K) is nonzero only for even L,K
!          X(L,K) = X(L,-K)  [similar to potential coefficients]

!          This code has been updated to include the possibility of a
!          nonaxial diffusion tensor (Rx .ne. Ry .ne. Rz). It was
!          developed from subroutine CALXLK by D.J. Schneider which was
!          based on the original subroutine matr written by G.Moro.


!       written by DEB 8-JUL-92

!       Includes:
!               nlsdim.inc
!               eprprm.inc
!               rndoff.inc

!       Uses:
!               w3j.f

!**********************************************************************

    subroutine anxlk(ipt,lptmx,kptmx,cpot,rp,rm,rz,xlk)
    implicit none

!    include 'limits.inc'
!    include 'simparm.inc'
    include 'rndoff.inc'

    double precision :: rp,rm,rz

    integer :: k,kx,kptf,k1f,k1i,k1,k1abs,k1x,k2,k2abs,k2f,k2i,k2x, &
    l,lx,l1,l1x,l2,l2x
    double precision :: factl,wig1,rj2u,rjuju,term

    double precision :: w3j
    external w3j

!######################################################################
!   new parameters added
    double precision :: xlk(5,5), cpot(5,5)
    integer          :: ipt, lptmx, kptmx
   
!    write(*,*) 'anxlk: cpot(2,1)=',cpot(2,1)
!    write(*,*) 'lptmx, kptmx:', lptmx, kptmx
!    write(*,*) 'rp,rm,rz:', rp,rm,rz

    do 20 lx=1,5
        do 10 kx=1,5
            xlk(lx,kx)=0.0D0
        10 END DO
    20 END DO

!----------------------------------------------------------------------
!     exit if no potential
!----------------------------------------------------------------------
!    write(*,*) 'ipt=',ipt
    if(ipt == 0) return

!---------------------------------------------------------------------
!     calculate xlk coefficients
!----------------------------------------------------------------------

! --- loop over L

    do 100 l=0,lptmx*2,2
        lx=1+l/2
        factl=dble(2*l+1)
        kptf=min(l,2*kptmx+2)
    
    ! --- loop over K ---
    
        do 90 k=0,kptf,2
            kx=1+k/2
        
        !------------------------------
        !         (J*R*J)U term
        !------------------------------
        
            rj2u=cpot(lx,kx)*( rp*dble(l*(l+1)-k*k) + rz*dble(k*k) )
            if (k+2 <= l .AND. k+2 <= kptmx) rj2u=rj2u+rm*cpot(lx,kx+1) &
            * dsqrt(dble((l+k+1)*(l+k+2)*(l-k-1)*(l-k) ) )
            if (k-2 >= 0) rj2u=rj2u+rm*cpot(lx,kx-1) &
            * dsqrt(dble((l-k+1)*(l-k+2)*(l+k-1)*(l+k) ) )
            if (k-2 < 0) rj2u=rj2u+rm*cpot(lx,kx+1) &
            * dsqrt(dble((l-k+1)*(l-k+2)*(l+k-1)*(l+k) ) )
            xlk(lx,kx)=-0.5d0*rj2u
        
        !------------------------------
        !      (JU)*R*(JU) term
        !------------------------------
        
            rjuju = 0.0d0
        
        !      --- loop over L1
        
            do 80 l1=0,lptmx,2
                l1x=1+l1/2
                k1f=min(l1,kptmx)
                k1i=-k1f
            
            !        --- loop over L2
            
                do 70 l2=0,lptmx,2
                    l2x=1+l2/2
                    if(l1+l2 >= l) then
                        wig1=w3j(l1,l,l2,0,0,0)
                    else
                        wig1=0.0D0
                        go to 70
                    end if
                
                !         --- loop over K1
                
                    do 60 k1=k1i,k1f,2
                        k1abs=abs(k1)
                        k1x=1+k1abs/2
                    
                    !        ---- loop over K2
                    
                        k2i=max(k-k1-2,-kptmx,-l2)
                        k2f=min(k-k1+2,kptmx,l2)
                    
                        do 50 k2=k2i,k2f,2
                            k2abs=abs(k2)
                            k2x=1+k2abs/2
                        
                        !        ------- (J_+ U)(J_+ U) term
                        
                            if( (k2 == k-k1-2) .AND. (abs(k1+1) <= l1) &
                             .AND. (abs(k2+1) <= l2) ) then
                                term=rm*w3j(l1,l,l2,k1+1,-k,k2+1)* dsqrt( &
                                dble((l1-k1)*(l1+k1+1)*(l2-k2)*(l2+k2+1)) )
                            
                            !       ------- (J_- U)(J_- U) term
                            
                            else if ( (k2 == k-k1+2) .AND. (abs(k1-1) <= l1) &
                                 .AND. (abs(k2-1) <= l2) ) then
                                term=rm*w3j(l1,l,l2,k1-1,-k,k2-1)* dsqrt( &
                                dble((l1+k1)*(l1-k1+1)*(l2+k2)*(l2-k2+1)) )
                            
                            !      ------- (J_+ U)(J_- U) term
                            
                            else if (k2 == k-k1) then
                                if((abs(k1+1) <= l1) .AND. (abs(k2-1) <= l2)) then
                                    term=rp*w3j(l1,l,l2,k1+1,-k,k2-1)* &
                                    dsqrt(dble((l1-k1)*(l1+k1+1)*(l2+k2)*(l2-k2+1)))
    !                                write(*,*) 'at least reaching the right location in (J_+)(J_-)', rp,&
    !                                                                            w3j(l1,l,l2,k1+1,-k,k2-1),&
    !                                                                            dsqrt(dble((l1-k1)*(l1+k1+1)*(l2+k2)*(l2-k2+1)))
                                else
                                    term=0.0D0
    !                                write(*,*) 'term got set to 0 in anxlk'
                                end if
                            
                            !      ------ (Jz U)(Jz U) term
                            
                                if (k2abs <= l2) &
                                term=term+rz*dble(k1*k2)*w3j(l1,l,l2,k1,-k,k2)
                                                        
                            else
                                term=0.0d0
                            end if
                        
                            rjuju=rjuju+cpot(l1x,k1x)*cpot(l2x,k2x)*wig1*term
 !                           write(*,*) 'term=', term
                        50 END DO
                    60 END DO
                70 END DO
            80 END DO
            xlk(lx,kx)=xlk(lx,kx)-0.25D0*factl*rjuju
 !           write(*,*) 'anxlk: factl, rjuju=', lx, kx, factl, rjuju
        90 END DO
    100 END DO

    do 120 lx=1,5
        do 110 kx=1,5
            if (abs(xlk(lx,kx)) < rndoff) xlk(lx,kx)=0.0D0
        110 END DO
    120 END DO

    return
    end subroutine anxlk
