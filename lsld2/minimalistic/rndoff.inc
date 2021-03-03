!	VERSION	1.0	2/5/99
!*********************************************************************

!	declarations of	machine	epsilon	parameter

!	Notes:
!		1) The machine epsilon,	eps, is	defined	to be the
!		   largest number such that the	result of the
!		   expression

!			delta=(1.0D0+eps)-1.0D0

!		   returns the value delta=0.0D0.  In other words,
!		   1.0D0+eps is	smallest double	precision floating
!		   point number	greater	than 1.0D0.

!		2) A FORTRAN 77	program	to determine the machine
!		   epsilon is included below for completeness.
!		   This	program	should be compiled with	all
!		   compiler optimization disabled.  The	parameter
!		   rndoff declared in this file	should assigned	to
!		   be a	small multiple of the larger of	the two
!		   numbers printed by this program.  This program
!		   is an adaptation of the function subroutine epslon
!		   supplied with EISPACK.

!		3) On most DEC machines, the two methods of
!		   calculating the machine epsilon give	the nearly
!		   same	the result, eps=2.77555756D-17.	 An
!		   appropriate value of	rndoff is then 1.0D-16.	 An
!		   example of a	machine	which gives grossly
!		   different results is	the Prime 9955.

!	written	by DJS 11-SEP-87

!*********************************************************************
    double precision :: rndoff
    parameter (rndoff=3.0D-15)

!	program	meps

!	double precision a,b,c,eps

!#######################################################################

!	a=1.0D0
! 0	b=a+1.0D0
!	eps=c
!	c=b-1.0D0
!	if (c.eq.0.0D0)	go to 20
!	a=a/2.0D0
!	go to 10

! 0	write (*,*) 'first epsilon = ',eps

!	a=4.0D0/3.0D0
! 0	b=a-1.0D0
!	c=b+b+b
!	eps=dabs(c-1.0D0)
!	if (d.eq.0.0D0)	go to 30

!	write (*,*) 'second epsilon = ',eps

!	stop
!	end

!***********************************************************************


