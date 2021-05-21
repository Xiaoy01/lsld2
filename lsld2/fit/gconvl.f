c Version 1.3.2
c----------------------------------------------------------------------
c                    =========================
c                       subroutine GCONVL
c                    =========================
c
c     Convolute given real-valued spectrum with a Gaussian lineshape 
c     having a specified derivative peak-to-peak linewidth.
c
c     NOTE: this differs from EPRLL version, which transforms a
c     complex-valued spectrum.      
c
c----------------------------------------------------------------------
      subroutine gconvl( spectr,wline,dfld,nfld,nft, tmpdat)
!f2py intent(in) spectr
!f2py intent(in) wline
!f2py intent(in) dfld
!f2py intent(in) nfld
!f2py intent(in) nft
!f2py intent(out) tmpdat

      implicit none
c
      integer nfld,nft
c
      integer i,k,no2
      double precision df,f,g,gnorm
c
c      include 'nlsdim.inc'
c      include 'ftwork.inc'
c      include 'pidef.inc'
c
      double precision EIGHT,EPS,ONE,TWO,THIRD,ZERO
      double precision PI,RADIAN
      integer MXPT
      parameter ( EIGHT=8.0D0,ONE=1.0D0,TWO=2.0D0,ZERO=0.0D0,
     #            EPS=1.0D-6,THIRD=0.33333333333333D0 )
      parameter (PI=3.1415926535897932384D0,RADIAN=PI/180.0D0)
      parameter (MXPT=4096)
      double precision tmpdat(MXPT)
      double precision spectr(MXPT),wline,dfld
c
c######################################################################
c
      if (wline.lt.EPS) return
c
c     -----------------------------------------------
c     Store calculation in tmpdat, zero-pad, and FFT
c     -----------------------------------------------
      do i=1,nfld
         tmpdat(i)=spectr(i)
      end do
c
      do i=nfld+1,nft
         tmpdat(i)=ZERO
      end do
c
      no2=nft/2
      call realft( tmpdat,no2,+1 )
c
c     ------------------------------------------------------------
c     Convolute with Gaussian function by multiplying in
c     with a Gaussian in Fourier space.
c
c     NOTE: REALFT returns only the positive frequencies
c     since FT of a real function is symmetric. Also, the
c     first and last elements of the FT array are real-valued
c     and returned as the real and imaginary parts of tmpdat(1)
c     ------------------------------------------------------------
      df=TWO*PI/(nft*dfld)
      gnorm=ONE/dfloat(no2)
      tmpdat(1)=tmpdat(1)*gnorm
      tmpdat(2)=tmpdat(2)*gnorm*dexp(-(no2*df*wline)**2/EIGHT)
      f=df
      do i=3,nft,2
         g=gnorm*dexp( -(f*wline)**2/EIGHT )
         tmpdat(i)=tmpdat(i)*g
         tmpdat(i+1)=tmpdat(i+1)*g
         f=f+df
      end do
c
c     ------------------------------------------------------------
c     Back-transfor to obtain convoluted spectrum and restore in
c     spectr array
c     ------------------------------------------------------------
      call realft( tmpdat,no2,-1 )
c no need, as tmpdat is my output now 
c      do i=1,nfld
c         spectr(i)=tmpdat(i)
c      end do
c
      return
      end


      subroutine realft(data,n,isign)
      implicit none
      integer i,i1,i2,i3,i4,isign,n,n2p3
      double precision c1,c2,h1r,h1i,h2r,h2i,wrs,wis,wr,wi,wpr,wpi,
     #                 wtemp,theta
      double precision data(2*n)
c
      theta=6.28318530717959d0/2.0d0/dble(n)
      c1=0.5d0
      if (isign.eq.1) then
        c2=-0.5d0
        call four1(data,n,+1)
      else
        c2=0.5d0
        theta=-theta
      endif
      wpr=-2.0d0*dsin(0.5d0*theta)**2
      wpi=dsin(theta)
      wr=1.d0+wpr
      wi=wpi
      n2p3=2*n+3
      do 11 i=2,n/2+1
        i1=2*i-1
        i2=i1+1
        i3=n2p3-i2
        i4=i3+1
        wrs=sngl(wr)
        wis=sngl(wi)
        h1r=c1*(data(i1)+data(i3))
        h1i=c1*(data(i2)-data(i4))
        h2r=-c2*(data(i2)+data(i4))
        h2i=c2*(data(i1)-data(i3))
        data(i1)=h1r+wrs*h2r-wis*h2i
        data(i2)=h1i+wrs*h2i+wis*h2r
        data(i3)=h1r-wrs*h2r+wis*h2i
        data(i4)=-h1i+wrs*h2i+wis*h2r
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
11    continue
      if (isign.eq.1) then
        h1r=data(1)
        data(1)=h1r+data(2)
        data(2)=h1r-data(2)
      else
        h1r=data(1)
        data(1)=c1*(h1r+data(2))
        data(2)=c1*(h1r-data(2))
        call four1(data,n,-1)
      endif
      return
      end

      subroutine four1(data,nn,isign)
      implicit none
      integer i,istep,j,m,mmax,n,nn,isign
      double precision wr,wi,wpr,wpi,wtemp,tempi,tempr,theta
      double precision data(2*nn)
c
      n=2*nn
      j=1
      do 11 i=1,n,2
        if(j.gt.i)then
          tempr=data(j)
          tempi=data(j+1)
          data(j)=data(i)
          data(j+1)=data(i+1)
          data(i)=tempr
          data(i+1)=tempi
        endif
        m=n/2
1       if ((m.ge.2).and.(j.gt.m)) then
          j=j-m
          m=m/2
        go to 1
        endif
        j=j+m
11    continue
      mmax=2
2     if (n.gt.mmax) then
        istep=2*mmax
        theta=6.28318530717959d0/(isign*mmax)
        wpr=-2.d0*dsin(0.5d0*theta)**2
        wpi=dsin(theta)
        wr=1.d0
        wi=0.d0
        do 13 m=1,mmax,2
          do 12 i=m,n,istep
            j=i+mmax
            tempr=sngl(wr)*data(j)-sngl(wi)*data(j+1)
            tempi=sngl(wr)*data(j+1)+sngl(wi)*data(j)
            data(j)=data(i)-tempr
            data(j+1)=data(i+1)-tempi
            data(i)=data(i)+tempr
            data(i+1)=data(i+1)+tempi
12        continue
          wtemp=wr
          wr=wr*wpr-wi*wpi+wr
          wi=wi*wpr+wtemp*wpi+wi
13      continue
        mmax=istep
      go to 2
      endif
      return
      end


