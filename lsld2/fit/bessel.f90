!  VERSION 1.0     2/5/99
!*********************************************************************

!               MODIFIED BESSEL FUNCTIONS OF THE FIRST
!               KIND OF INTEGER ORDER AND REAL ARGUMENT
!               ---------------------------------------

!       This double precision function subroutine calculates the
!       value of the modified Bessel function of the first kind of
!       integer order and strictly real argument via a backward
!       recurrence scheme taken from "Numerical Recipes", W. H. Press,
!       B. P. Flannery, S. A. Teulosky and W. T. Vetterling, 1st ed.,
!       Cambridge Univ. Press, 1986.

!       written by DJS 10-SEP-87
!       Bug in small-argument Taylor series expansion fixed by DEB OCT-92


!       Includes:
!               rndoff.inc

!       Uses:
!               bessi0.f
!               bessi1.f

!*********************************************************************

    function bessi(n,z)
    implicit none
    integer :: n
    double precision :: bessi,z

    include 'rndoff.inc'

    double precision :: bessi0,bessi1
    external bessi0,bessi1

    integer :: iacc
    double precision :: bigno,bigni
    parameter (iacc=40,bigno=1.0D10,bigni=1.0D-10)

    integer :: i,m,mmax
    double precision :: x,phase,twobyx,bi,bip,bim

    intrinsic abs

!#####################################################################

!---------------------------------------------------------------------
!       get proper phase factor if argument is negative with the
!       following rules
!                                         n
!       I (z) = I (z)  and   I (-z) = (-1) I (z)
!        n       -n           n             n
!---------------------------------------------------------------------

    m=abs(n)
    x=abs(z)

    if ((z < 0.0D0) .AND. (mod(m,2) == 1)) then
        phase=-1.0D0
    else
        phase=1.0D0
    end if

!---------------------------------------------------------------------
!       return proper values if argument is zero
!---------------------------------------------------------------------

    if (x < rndoff) then
        if (m == 0) then
            bessi=1.0D0
        else
            bessi=0.0D0
        end if
        return
    end if

!---------------------------------------------------------------------
!       call bessi0 if n=0, bessi1 if n=1, or go through
!       downward recurrence if n>1.
!---------------------------------------------------------------------

    if (m == 0) then
        bessi=phase*bessi0(x)
    else if (m == 1) then
        bessi=phase*bessi1(x)
    else
        bessi=0.0D0
        twobyx=2.0D0/x
        bip=0.0D0
        bi=1.0D0
        mmax=2*((m+int(sqrt(dble(iacc*m)))))
        do 10 i=mmax,1,-1
            bim=bip+dble(i)*twobyx*bi
            bip=bi
            bi=bim
            if (abs(bi) > bigno) then
                bessi=bessi*bigni
                bi=bi*bigni
                bip=bip*bigni
            end if
            if (i == m) bessi=bip
        10 END DO
        bessi=phase*bessi*bessi0(x)/bi
    end if

    return
    end function bessi

!*********************************************************************

!               MODIFIED BESSEL FUNCTION OF THE FIRST
!               KIND OF ORDER ZERO AND REAL ARGUMENT
!               -------------------------------------

!       This double precision function subroutine calculates the
!       modified Bessel function of the first kind of order zero
!       and real argument by either the Taylor series expansion
!       for small arguments or the first term of the asymptotic
!       series for sufficiently large arguments.

!       written by DJS 10-SEP-87

!       Includes:
!               rndoff.inc
!               pidef.inc

!       Uses:

!*********************************************************************

    function bessi0(z)

    include 'rndoff.inc'
!    include 'pidef.inc'
    double precision :: pi,radian
    parameter (pi=3.1415926535897932384D0,radian=pi/180.0D0)


    double precision :: bessi0
    double precision :: z

    integer :: i,j
    double precision :: x,y,smax,temp1,temp2,temp3,sum

    integer :: nmax
    parameter (nmax=40)

    double precision :: tser
    dimension tser(nmax)

    double precision :: cutoff
    parameter (cutoff=20.0D0)

!######################################################################

    y=abs(z)

!------------------------------------------------------------
!     Set function value to unity if argument is too small
!------------------------------------------------------------
    if (y < rndoff) then
        bessi0=1.0D0
    
    !-------------------------------------------------------------
    !     Taylor series expansion for small to moderate arguments
    !-------------------------------------------------------------
    else if (y <= cutoff) then
        x=y*y*0.25D0
        temp1=1.0D0
        smax=1.0D0
        i=1
        10 temp1=(temp1/dble(i))*(x/dble(i))
        if (i > nmax) then
            write(*,1000)
            stop
        end if
        tser(i)=temp1
        i=i+1
        if (temp1 > smax) smax=temp1
        if (temp1/smax > rndoff) go to 10
    
        bessi0=0.0D0
        do 20 j=i-1,1,-1
            bessi0=bessi0+tser(j)
        20 END DO
        bessi0=bessi0+1.0D0
    
    !----------------------------------------------
    !     Asymptotic expansion for large arguments
    !----------------------------------------------
    else
        x=0.125D0/y
        sum=0.0D0
        temp3=1.0D0
        smax=1.0D0
        i=1
        30 temp1=dble(2*i-1)
        temp2=(x*temp1)*(temp1/dble(i))
        if (temp2 > 1.0D0) go to 40
        temp3=temp3*temp2
        if (temp3 > smax) smax=temp3
        if (temp3/smax < rndoff) go to 40
        sum=sum+temp3
        i=i+1
        go to 30
        40 bessi0=dexp(y)*((sum+1.0D0)/dsqrt(y*(pi+pi)))
    end if

    return

    1000 format('bessi0: Taylor series did not converge')
    end function bessi0

!*********************************************************************

!               MODIFIED BESSEL FUNCTION OF THE FIRST
!               KIND OF ORDER ONE AND REAL ARGUMENT
!               -------------------------------------

!       This double precision function subroutine calculates the
!       modified Bessel function of the first kind of order one
!       and real argument by either the Taylor series expansion
!       for small arguments or the first term of the asymptotic
!       series for sufficiently large arguments.

!       written by DJS 10-SEP-87

!       Includes:
!               rndoff.inc
!               pidef.inc

!       Uses:

!*********************************************************************

    function bessi1(z)

    include 'rndoff.inc'
!    include 'pidef.inc'
    double precision :: pi,radian
    parameter (pi=3.1415926535897932384D0,radian=pi/180.0D0)

    double precision :: bessi1
    double precision :: z

    integer :: i,j
    double precision :: x,y,smax,temp1,temp2,temp3,phase,sum

    integer :: nmax
    parameter (nmax=40)

    double precision :: series
    dimension series(nmax)

    double precision :: cutoff
    parameter (cutoff=20.0D0)

!#####################################################################

    if (z > 0.0D0) then
        phase=1.0D0
        y=z
    else
        phase=-1.0D0
        y=-z
    end if

!----------------------------------------------------------------------
!     set answer to zero if argument is too small, otherwise
!----------------------------------------------------------------------
    if (y < rndoff) then
        bessi1=0.0D0
    
    !----------------------------------------------------------------------
    !     Use Taylor series expansion for small to moderate arguments or
    !----------------------------------------------------------------------
    else if (y <= cutoff) then
        x=y*y*0.25D0
        temp1=1.0D0
        smax=1.0D0
        i=1
        10 temp1=(temp1/dble(i))*(x/dble(i+1))
        if (i > nmax) then
            write(*,1000)
            stop
        end if
        series(i)=temp1
        i=i+1
        if (temp1 > smax) smax=temp1
        if (temp1/smax > rndoff) go to 10
        bessi1=0.0D0
        do 20 j=i-1,1,-1
            bessi1=bessi1+series(j)
        20 END DO
        bessi1=phase*y*0.5D0*(bessi1+1.0D0)
    
    !----------------------------------------------------------------------
    !     asymptotic expansion for large arguments
    !----------------------------------------------------------------------
    else
        x=0.125D0/y
        sum=3.0D0*x
        temp3=sum
        smax=1.0D0
        i=2
        30 temp1=dble(2*i-1)
        temp1=temp1*temp1-4.0D0
        temp2=(x*temp1)/dble(i)
        if (temp2 > 1.0D0) go to 40
        temp3=temp3*temp2
        if (temp3 > smax) smax=temp3
        if (temp3/smax < rndoff) go to 40
        sum=sum+temp3
        i=i+1
        go to 30
        40 bessi1=dexp(y)*(1.0D0-sum)/dsqrt(y*(pi+pi))
    end if

    return

!----------------------------------------------------------------------
    1000 format('bessi0: Taylor series did not converge')
    end function bessi1
