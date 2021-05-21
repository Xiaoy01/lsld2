! NLSPMC Version 1.0 2/5/99
!----------------------------------------------------------------------
! This file contains two copies of subroutine named CCRINT and CCRIN1
! for use with double integration problems (e.g. integration over
! the polar angles in rigid limit EPR calculation programs)
!----------------------------------------------------------------------

!                    ======================
!                      subroutine CCRINT
!                    ======================

!  Clenshaw-Curtis-Romberg integration method according to algorithm
!  of O'Hara and Smith (Computer J. 12 (1969) 179-82)
!  Adapted from G. Bruno thesis, Cornell Univ. (see p. 554 Bruno thesis)
!  Code rearranged clarified, and re-commented by D. Budil, 1/16/90

!  The function to be integrated is supplied as a double precision
!  function whose name is passed to the program (external function f)

!   Parameters:
!       bndlow          Lower bound of integration
!       bndhi           Upper bound of integration
!       epsiln          Tolerance (absolute) for convergence of integral
!			Relative tolerance ratio is taken to be epsiln/100
!       small           Smallest permissible step, as fraction of (x range)/4

!   Returns:
!       neval           Number of function evaluations
!       id              Error flag (1 for normal, -1 for stack overflow)
!----------------------------------------------------------------------
    subroutine ccrint_new(bndlow,bndhi,epsiln,small,sum,neval,id,lr,kr,ipt,cpot,lptmx,kptmx)
!    external f

    integer :: id, neval, nstack
    double precision :: b1(30), b2(30), b3(30), b4(30), p(30)
    double precision :: bndlow, bndhi, acl7, ainsc, ccerr, &
    cnc1, cnc2, cnc3, cnc4, c7l1, c7l2, c7l3, c7l4, &
    c8r1, c8r2, c8r3, c8r4, &
    cvgtol, err, diff, h, hmin, rombrg, rt3, &
    sa, sc, sd, small, sum, &
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, epsiln, &
    x0, x1, x2, x3, x4, x5, x6, x7, x8, xx
    double precision :: dabs 
    double precision :: ans, ans1

    integer :: ipt, lptmx,kptmx,lr,kr
    double precision cpot(5,5)
! ======================================================================
! Coefficients for 2*(5 point Newton-Cotes) integration
! 14/45, 28/45, 24/45, 64/45
! ======================================================================

    data cnc1 /0.3111111111111111d0 / &
    cnc2 /0.6222222222222222d0 / &
    cnc3 /0.5333333333333333d0 / &
    cnc4 /1.4222222222222222d0 /

! ======================================================================
! Coefficients for 8 point Romberg formula
! 868/2835, 1744/2835, 1408/2835, 4096/2835
! ======================================================================

    data c8r1 /0.3061728395061728d0 / &
    c8r2 /0.6151675485008818d0 / &
    c8r3 /0.4966490299823633d0 / &
    c8r4 /1.444797178130511d0  /

! ======================================================================
! Coefficients for 7 point Clenshaw-Curtis formula
! 36/315, 656/315, 576/315, 320/315
! ======================================================================

    data c7l1 /0.1142857142857143d0 / &
    c7l2 /2.082539682539683d0  / &
    c7l3 /1.828571428571429d0  / &
    c7l4 /1.015873015873016d0  /

    data rt3  /1.732050807568877d0 /

! ======================================================================
! Set initial domain to be the entire domain from lower to upper bound
! ======================================================================
    id = 1
    h  = (bndhi-bndlow)*0.25d0
    x0 = bndlow
    x2 = x0+h
    x4 = x2+h
    x6 = x4+h
    x8 = bndhi

!    write(*,*) 'started in ccrint_new, about to call fz'
    call fz(x0, lr,kr,lptmx,kptmx,cpot, ans)
!    write(*,*) 'started in ccrint_new, called fz'
    t0 = ans    !    t0 = f(x0)
    call fz(x2, lr,kr,lptmx,kptmx,cpot, ans)
    t2 = ans    !    t2 = f(x2)
    call fz(x4, lr,kr,lptmx,kptmx,cpot, ans)
    t4 = ans    !    t4 = f(x4)
    call fz(x6, lr,kr,lptmx,kptmx,cpot, ans)
    t6 = ans    !    t6 = f(x6)
    call fz(x8, lr,kr,lptmx,kptmx,cpot, ans)
    t8 = ans    !    t8 = f(x8)

    neval = 5

    err    = epsiln
    sum    = 0.0d0
    nstack = 0
    hmin   = small*h

! ======================================================================
! Beginning of an iteration: divide step size in half.
! (assumed that current domain is only a small fraction of the total)
! Supplied with a (sub)domain defined by the x values x0,x2,x4,x6,x8,
! interpolate the points x1, x3, x5, x7 and calculate f at these points
! ======================================================================

    4 h = 0.5d0*h
    if (dabs(h) <= dabs(hmin)) go to 9

    x1 = x0+h
    x3 = x2+h
    x5 = x4+h
    x7 = x6+h
!    write(*,*) 'in ccrint_new, about to call fz'
    call fz(x1, lr,kr,lptmx,kptmx,cpot, ans)
    t1 = ans    !    t0 = f(x1)
    call fz(x3, lr,kr,lptmx,kptmx,cpot, ans)
    t3 = ans    !    t3 = f(x3)
    call fz(x5, lr,kr,lptmx,kptmx,cpot, ans)
    t5 = ans    !    t5 = f(x5)
    call fz(x7, lr,kr,lptmx,kptmx,cpot, ans)
    t7 = ans    !    t7 = f(x7)

    neval=neval+4

    cvgtol = err*0.1D0
    if (nstack == 0) cvgtol = err

! ======================================================================
! First convergence test:
! Compare results of Newton-Cotes and Romberg quadratures.
! If absolute AND relative differences are too large, the current domain
! must be subdivided

! *** Modified 22-May-90: proceed if the integrated function is zero -DEB
! ======================================================================

    sa=t0+t8
    sc=t2+t6
    sd=t1+t3+t5+t7

    rombrg = (c8r1*sa + c8r2*t4 +c8r3*sc + c8r4*sd)*h
    ainsc  = (cnc1*sa + cnc2*t4 +cnc3*sc + cnc4*sd)*h
    diff    = dabs(rombrg-ainsc)

    if ( rombrg /= 0.0D0 .AND. ainsc /= 0.0D0) then
        if ( diff > cvgtol &
         .AND. diff/dabs( 0.5D0*(rombrg+ainsc) ) > cvgtol*1.0D-2 ) &
        go to 10
    endif

! ======================================================================
! Second convergence test:
! Compare 8 pt. Romberg with 7 pt. Clenshaw-Curtis quadratures.
! (Use error estimate for Clenshaw-Curtis quadrature if it is larger)
! If absolute AND relative differences are too large, the current domain
! needs to be subdivided

! *** Modified 22-May-90: proceed if the integrated function is zero -DEB
! ======================================================================

    xx  = rt3*h
    call fz(x2-xx, lr,kr,lptmx,kptmx,cpot, ans)
    call fz(x2+xx, lr,kr,lptmx,kptmx,cpot, ans1)
    t9 = ans + ans1 ! t9  = f(x2-xx) + f(x2+xx)
    call fz(x6-xx, lr,kr,lptmx,kptmx,cpot, ans)
    call fz(x6+xx, lr,kr,lptmx,kptmx,cpot, ans1)
    t10 = ans + ans1 ! t10 = f(x6-xx) + f(x6+xx)
    neval=neval+4
    acl7 = 0.5D0*h*(  c7l1*(sa+2.d0*t4) + c7l2*sc + c7l3*sd &
    + c7l4*(t9+t10) )
    diff  = dabs( acl7-rombrg )
    ccerr = (  dabs(0.5d0*(t0+t4)+t1+t3-t2-t9 ) &
    + dabs(0.5d0*(t4+t8)+t5+t7-t6-t10) )*h*64.d0/945.d0
    if ( ccerr > diff ) diff = ccerr

    if ( rombrg /= 0.0D0 .AND. acl7 /= 0.0D0) then
        if ( diff > cvgtol &
         .AND. diff/dabs(0.5d0*(acl7+rombrg)) > cvgtol/1.0D2) goto 10
    endif

! ======================================================================
! Convergence achieved for this subdomain: add integral into result
! Replace current subdomain with the top subdomain on the stack for next
! iteration (Integration is complete if the stack is empty)
! ======================================================================

    sum = sum + acl7

    if (diff < cvgtol) err = err + diff

    9 continue

    if (nstack <= 0) return

    h =  p(nstack)
    t0 = t8
    t2 = b1(nstack)
    t4 = b2(nstack)
    t6 = b3(nstack)
    t8 = b4(nstack)
    x0 = x8
    x2 = x0+h
    x4 = x2+h
    x6 = x4+h
    x8 = x6+h
    nstack = nstack-1
    go to 4

! ======================================================================
! Push upper half of current subdomain (points x5, x6, x7, x8)
! and function values onto the stack (error return if stack overflows)
! Replace current subdomain (points x0, x2, x4, x6, x8) and function values
! with its lower half (points x0, x1, x2, x3, x4) for next iteration
! ======================================================================

    10 nstack=nstack+1
    if(nstack > 30) then
        id=-1
        write(6,*)'CCRINT: more than 30 subdomain divisions'
        return
    endif

    b1(nstack) = t5
    b2(nstack) = t6
    b3(nstack) = t7
    b4(nstack) = t8
    p(nstack)  = h

    t8 = t4
    t6 = t3
    t4 = t2
    t2 = t1
    x8 = x4
    x6 = x3
    x4 = x2
    x2 = x1
    go to 4
    end subroutine ccrint_new
