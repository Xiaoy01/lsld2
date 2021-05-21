! NLSMC VERSION 1.0   2/5/99
!**********************************************************************

!       This double precision function evaluates the integrand of
!       the orientational integral in the definition of the starting
!       vector.  The integral over the gamma Euler angle is done
!       analytically.

!       Notes:
!               1) The L and K quantum numbers are passed via a
!                  common block.
!               2) For more information on the evaluation of the
!                  modified Bessel and associated Legendre functions
!                  see the respective routines.
!               3) The often used special case of lptmx=2, kptmx=0
!                  is handled separately for faster execution.

!       written by DJS 10-SEP-87

!       Includes:
!               nlsdim.inc
!               eprprm.inc

!       Uses:
!               bessi.f
!               plgndr.f

!**********************************************************************


subroutine fz(z, lr,kr,lptmx,kptmx,cpot, ans)

!    double precision :: fz
    double precision :: z, ans

    include 'limits.inc'
!    include 'simparm.inc'

    integer :: lr,kr
!    common/ifzdat/lr,kr

    double precision :: a,b
    integer :: k

    double precision :: dsq24,dsq360
    parameter (dsq24=4.89897948556635619639, &
    dsq360=18.97366596101027599199)

    double precision :: bessi,plgndr
    external bessi,plgndr
    
    integer :: lptmx, kptmx
    double precision :: cpot(5,5)
!######################################################################

    if((lptmx == 2) .AND. (kptmx == 0)) then
        if(kr == 0) then
            ans=dexp(0.5D0*cpot(2,1)*plgndr(2,0,z))* &
            plgndr(lr,kr,z)
        else
            ans=0.0D0
        end if
    else
        a=0.5D0*(cpot(2,1)*plgndr(2,0,z) &
        +cpot(3,1)*plgndr(4,0,z))
        if (kptmx /= 0) then
            b=cpot(2,2)*plgndr(2,2,z)/dsq24 &
            +cpot(3,2)*plgndr(4,2,z)/dsq360
        else
            b=0.0D0
        end if
        k=kr/2
        ans=bessi(k,b)*dexp(a)*plgndr(lr,kr,z)
    end if

!----------------------------------------------------------------------
!     return to caller
!----------------------------------------------------------------------

    return
!    END PROGRAM
end subroutine fz

