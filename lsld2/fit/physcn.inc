! Physical constants (in cgs/electromagnetic units)
!    ge:     free-electron g-value
!    betae:  Bohr magneton
!    hbar:   Planck's constant
!    gammae: electronic gyromagnetic ratio
!    kb:     Boltzmann's constant
!----------------------------------------------------------------------
    double precision :: GE,BETAE,GAMMAE,HBAR,KB
    parameter( GE=2.002322D0, &
    BETAE=9.2731D-21, &
    HBAR=1.05443D-27, &
    GAMMAE=GE*BETAE/HBAR, &
    KB=1.380622D-16 )
