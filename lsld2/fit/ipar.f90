!       VERSION 1.0     2/5/99
!*********************************************************************

!                         SUBROUTINE IPAR
!                         ===============

!        This integer function subroutine returns the value -1
!        if its argument is odd, and +1 if it is even.

!*********************************************************************

    integer function ipar(num)

    integer :: num

!#####################################################################

    if (mod(num,2) == 0) then
        ipar=1
    else
        ipar=-1
    end if
    return
    end function ipar
