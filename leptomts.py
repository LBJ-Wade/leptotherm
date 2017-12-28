from scipy.special import kn
import numpy as np
from odeintw import odeintw

def my_kn1(x):
    """
    Convenience wrapper for kn(1, x)
    """
    return kn(1, x) if x<=600 else 1e-100

def my_kn2(x):
    """
    Convenience wrapper for kn(2, x)
    """
    return kn(2, x) if x<=600 else 1e-100


class LeptoCalc(object):
    def __init__(self, debug=False, nds=1, approx=False, xmin=1e-1, xmax=100, xsteps=1000, controlplots=False, plotprefix=""):
        """
        Set global constants here.
        """
        #Higgs vev in GeV
        self.v = 174.
        #relativistic degrees of freedom at high temperature
        self.gstar = 106.75
        #Planck mass in GeV
        self.MP = 1.22e+19
        #neutrino cosmological mass in GeV
        self.mstar = 1.0e-12
        self.debug=debug
        self.nds=nds
        self.approx=approx
        self._xmin = xmin
        self._xmax = xmax
        self._xsteps=xsteps
        self.controlplots=controlplots
        self.xs=None
        self.setXS()
        self.plotprefix=plotprefix


    def setXMin(self, x):
        self._xmin=x
        self.setXS()
    def setXMax(self, x):
        self._xmax=x
        self.setXS()
    def setXSteps(self, x):
        self._xsteps=x
        self.setXS()

    def setXS(self):
        self.xs = np.geomspace(self.xmin, self.xmax, self.xsteps)
        if self.debug:
            print("Integration range:",self.xs.min(),self.xs.max())


    @property
    def xmin(self):
        return self._xmin
    @property
    def xmax(self):
        return self._xmax
    @property
    def xsteps(self):
        return self._xsteps

    def setParams(self, pdict):
        """
        This set the model parameters. pdict is expected to be a dictionary
        """
        self.delta    = pdict['delta']/180*np.pi
        self.a        = pdict['a']/180*np.pi
        self.b        = pdict['b']/180*np.pi
        self.theta12  = pdict['theta12']/180*np.pi
        self.theta23  = pdict['theta23']/180*np.pi
        self.theta13  = pdict['theta13']/180*np.pi
        self.x1       = pdict['x1']/180*np.pi
        self.y1       = pdict['y1']/180*np.pi
        self.x2       = pdict['x2']/180*np.pi
        self.y2       = pdict['y2']/180*np.pi
        self.x3       = pdict['x3']/180*np.pi
        self.y3       = pdict['y3']/180*np.pi
        self.m1       = 10**pdict['m1'] * 1e-9 # NOTE input is in log10(m1) in eV --- we convert here to the real value in GeV
        self.M1       = 10**pdict['M1']  #
        self.M2       = 10**pdict['M2']  #
        self.M3       = 10**pdict['M3']  #
        self.ordering = pdict['ordering']

    # Some general calculators purely based on input parameters
    @property
    def R(self):
        """
        Orthogonal matrix R = R1.R2.R3
        """

        R1 = np.array([[1., 0., 0.],
                       [0.,  np.cos(self.x1+self.y1*1j), np.sin(self.x1+self.y1*1j)],
                       [0., -np.sin(self.x1+self.y1*1j), np.cos(self.x1+self.y1*1j)]], dtype=np.complex128)

        R2 = np.array([[ np.cos(self.x2+self.y2*1j), 0., np.sin(self.x2+self.y2*1j)],
                       [0., 1. , 0.],
                       [-np.sin(self.x2+self.y2*1j), 0., np.cos(self.x2+self.y2*1j)]], dtype=np.complex128)

        R3 = np.array([[ np.cos(self.x3+self.y3*1j), np.sin(self.x3+self.y3*1j), 0.],
                       [-np.sin(self.x3+self.y3*1j), np.cos(self.x3+self.y3*1j), 0.],
                       [0., 0., 1.]], dtype=np.complex128)

        return R1 @ R2 @ R3

    @property
    def SqrtDM(self):
        """
        Matrix square root of heavy masses
        """
        return np.array([[np.sqrt(self.M1), 0., 0.],
                         [0., np.sqrt(self.M2), 0.],
                         [0., 0., np.sqrt(self.M3)]], dtype=np.complex128)

    @property
    def DM(self):
        """
        Heavy mass matrix
        """
        return np.array([[self.M1, 0., 0.],
                         [0., self.M2, 0.],
                         [0., 0., self.M3]], dtype=np.complex128)

    @property
    def SqrtDm(self):
        """
        Matrix square root of light masses.
        Eveything is in GeV.
        """
        msplit2_solar       =  7.40e-5*1e-18 # 2017
        msplit2_athm_normal = 2.515e-3*1e-18 # Values
        msplit2_athm_invert = 2.483e-3*1e-18 # from nu-fit 3.1

        if self.ordering==0:
            M11 = np.sqrt(self.m1)
            M22 = np.sqrt(np.sqrt(msplit2_solar       + self.m1*self.m1))
            M33 = np.sqrt(np.sqrt(msplit2_athm_normal + self.m1*self.m1))
        elif self.ordering==1:
            M11 = np.sqrt(np.sqrt(msplit2_athm_invert + self.m1*self.m1 - msplit2_solar))
            M22 = np.sqrt(np.sqrt(msplit2_athm_invert + self.m1*self.m1))
            M33 = np.sqrt(self.m1)
        else:
            raise Exception("ordering %i not implemented"%self.ordering)

        return np.array([ [M11,  0.,  0.],
                          [ 0., M22,  0.],
                          [ 0.,  0., M33] ], dtype=np.complex128)

    @property
    def U(self):
        """
        PMNS
        """
        s12     = np.sin(self.theta12)
        s23     = np.sin(self.theta23)
        s13     = np.sin(self.theta13)
        c12     = np.power(1-s12*s12,0.5)
        c23     = np.power(1-s23*s23,0.5)
        c13     = np.power(1-s13*s13,0.5)
        return np.array([ [c12*c13,c13*s12*np.exp(self.a*1j/2.), s13*np.exp(self.b*1j/2-self.delta*1j)],
                           [-c23*s12 - c12*np.exp(self.delta*1j)*s13*s23,np.exp((self.a*1j)/2.)*(c12*c23 - np.exp(self.delta*1j)*s12*s13*s23) , c13*np.exp((self.b*1j)/2.)*s23],
                           [-c12*c23*np.exp(self.delta*1j)*s13 + s12*s23,np.exp((self.a*1j)/2.)*(-c23*np.exp(self.delta*1j)*s12*s13 - c12*s23) ,c13*c23*np.exp((self.b*1j)/2.)]], dtype=np.complex128)

    @property
    def h(self):
        """
        Yukawa matrices
        """
        return (1./self.v)*(self.U @ self.SqrtDm @ np.transpose(self.R) @ self.SqrtDM)

    @property
    def meff1(self):
        """
        Effective mass 1
        """
        return np.dot(np.conjugate(np.transpose(self.h)),self.h)[0,0]*(self.v**2)/self.M1


    @property
    def meff2(self,):
        """
        Effective mass 2
        """
        return np.dot(np.conjugate(np.transpose(self.h)),self.h)[1,1]*(self.v**2)/self.M2

    @property
    def k1(self):
        """
        Decay parameter 1
        """
        return self.meff1/self.mstar

    @property
    def k2(self):
        """
        Decay parameter 2
        """
        return self.meff2/self.mstar

    def D1(self, k1, z):
        """
        Decay term for Boltzmann equation
        """
        return  k1*z*my_kn1( z)/my_kn2( z)

    def KNR(self, x):
        """
        Test function to see the numerical stability behaviour of the ratio
        of two modified Bessel functions.
        """
        return my_kn1(x)/my_kn2(x)

    def D2(self, k,z):
        """
        Decay term for Boltzmann equation
        """
        r =self.M2/self.M1
        a = r*r
        x=np.real(r*z)
        b = my_kn1(x)
        c = my_kn2(x)
        return k*z*a*b/c

    def N1Eq(self, z):
        """
        Equilibrium N1 number density
        """
        n1 = 3./8.*(z**2)*my_kn2(z)
        return n1

    def N2Eq(self, z):
        """
        Equilibrium N2 number density
        For numerical reasons, cut off the return value if there are more than 5 orders of
        magnitude between N1Eq and N2Eq.
        """
        r = self.M2/self.M1
        n2 = 3./8.*np.power(r*z,2)*my_kn2(r*z)

        return n2

    def W1(self, k1, z):
        """
        Washout parameter 1
        """
        w1 = 1./4*(z**3)*k1*my_kn1(z)
        return w1

    def W2(self, k, z):
        """
        Washout parameter 2
        """
        w1=self.W1(k,z)
        r = self.M2/self.M1
        w2 = k*r/4*np.power(r*z,3) * my_kn1(r*z)
        return w2

    def W2p(self, k, z):
        """
        Washout parameter 2 --- derivative w.r.t. r*z
        """
        r = self.M2/self.M1
        w2p = k*r/4 *(3* np.power(r*z,2)*my_kn1(r*z) + np.power(r*z,3)*kvp(1,r*z))
        return w2p

    def hterm(self, a, b):
        """
        Probability coefficient

        a ... [0,1,2]
              0 = e
              1 = mu
              2 = tau
        b ... [0,1]
              0 = term1
              1 = term2

        """
        norm          = 1./((np.dot(np.conjugate(np.transpose(self.h)), self.h))[0,0])
        return norm*(np.abs(self.h[a,b])**2)

    def c1a(self, a):
        """
        Probability coefficient for 1 a
        """
        norm          = np.sqrt(1./((np.dot(np.conjugate(np.transpose(self.h)), self.h))[0,0]))
        return norm*(self.h[a,0])

    def c2a(self, a):
        """
        Probability coefficient for 1 a
        """
        norm          = np.sqrt(1./((np.dot(np.conjugate(np.transpose(self.h)), self.h))[1,1]))
        return norm*(self.h[a,1])

    def f1(self, x):
        """
        f1(x) appears in the expression for epsilon but is numerically unstable so approx. with pw definition
        """
        r2=np.power(x,2)

        f1temp = 2./3.*r2*( (1.+r2) * np.log( (1.+r2) / r2 ) - (2.-r2)/(1.-r2) )

        return f1temp if x<10000 else 1

    def f2(self, x):
        """
        f2(x) appears in the expression for epsilon
        """
        return (2./3.)*(1/(np.power(x,2)-1.))

    def epsilon(self, i, j, k, m):
        """
        CP asymmetry parameter
        """
        l         = self.h
        ldag      = np.conjugate(np.transpose(l))
        lcon      = np.conjugate(l)
        M         = self.DM
        lsquare   = np.dot(ldag,l)

        #define terms of epsilon: prefactor and first term (first), second term (second) etc.
        prefactor   = (3/(16*np.pi))*(1/(lsquare[i,i]))
        first       = np.imag(lsquare[i,j]*l[m,j]*lcon[m,i])*(M[i,i]/M[j,j])*self.f1(M[j,j]/M[i,i])

        second      = np.imag(lsquare[j,i]*l[m,j]*lcon[m,i])*(2./3.)*(1/(np.power(M[j,j]/M[i,i],2)-1.))
        third       = np.imag(lsquare[i,k]*l[m,k]*lcon[m,i])*(M[i,i]/M[k,k])*self.f1(M[k,k]/M[i,i])
        fourth      = np.imag(lsquare[k,i]*l[m,k]*lcon[m,i])*(2./3.)*(1/(np.power(M[k,k]/M[i,i],2)-1.))
        return prefactor*(first+second+third+fourth)

    def epsilonab(self, a, b):
        """
        CP asymmetry parameter. a and b are NOT the model parameters of the same name
        """
        l         = self.h
        ldag      = np.conjugate(np.transpose(l))
        lcon      = np.conjugate(l)
        M         = self.DM
        lsquare   = np.dot(ldag,l)

        #define terms of epsilon: prefactor and first term (first), second term (second) etc.
        prefactor   = (3/(32*np.pi))*(1/(lsquare[0,0]))
        first       = 1j*(lsquare[1,0]*l[a,0]*lcon[b,1]-lsquare[0,1]*l[a,1]*lcon[b,0]) * (M[0,0]/M[1,1])*self.f1(M[1,1]/M[0,0])
        third       = 1j*(lsquare[2,0]*l[a,0]*lcon[b,2]-lsquare[0,2]*l[a,2]*lcon[b,0])*(M[0,0]/M[2,2])*self.f1(M[2,2]/M[0,0])
        second      = 1j*(2./3.)*(1/(np.power(M[1,1]/M[0,0],2)-1.))*(l[a,0]*lcon[b,1]*lsquare[0,1]-lcon[b,0]*l[a,1]*lsquare[1,0])
        fourth      = 1j*(2./3.)*(1/(np.power(M[2,2]/M[0,0],2)-1.))*(l[a,0]*lcon[b,2]*lsquare[0,2]-lcon[b,0]*l[a,2]*lsquare[2,0])
        # print(a,b,prefactor)
        return prefactor*(first+second+third+fourth)

    def epsilon1ab(self,a,b):
        l         = self.h
        ldag      = np.conjugate(np.transpose(l))
        lcon      = np.conjugate(l)
        M         = self.DM
        lsquare   = np.dot(ldag,l)

        #define terms of epsilon: prefactor and first term (first), second term (second) etc.
        prefactor   = (3/(32*np.pi))*(1/(lsquare[0,0]))
        first       = 1j*(lsquare[1,0]*l[a,0]*lcon[b,1]-lsquare[0,1]*l[a,1]*lcon[b,0])*(M[0,0]/M[1,1])*self.f1(M[1,1]/M[0,0])
        third       = 1j*(lsquare[2,0]*l[a,0]*lcon[b,2]-lsquare[0,2]*l[a,2]*lcon[b,0])*(M[0,0]/M[2,2])*self.f1(M[2,2]/M[0,0])
        second      = 1j*(2./3.)*(1/(np.power(M[1,1]/M[0,0],2)-1.))*(l[a,0]*lcon[b,1]*lsquare[0,1]-lcon[b,0]*l[a,1]*lsquare[1,0])
        fourth      = 1j*(2./3.)*(1/(np.power(M[2,2]/M[0,0],2)-1.))*(l[a,0]*lcon[b,2]*lsquare[0,2]-lcon[b,0]*l[a,2]*lsquare[2,0])
        epsilon1abtemp = prefactor*(first+second+third+fourth)
        return epsilon1abtemp

    #CP asymmetry parameter
    def epsilon2ab(self,a,b):
        l         = self.h
        ldag      = np.conjugate(np.transpose(l))
        lcon      = np.conjugate(l)
        M         = self.DM
        lsquare   = np.dot(ldag,l)

        #define terms of epsilon: prefactor and first term (first), second term (second) etc.
        prefactor   = (3/(32*np.pi))*(1/(lsquare[1,1]))
        first       = 1j*(lsquare[0,1]*l[a,1]*lcon[b,0]-lsquare[1,0]*l[a,0]*lcon[b,1])*(M[1,1]/M[0,0])*self.f1(M[0,0]/M[1,1])
        third       = 1j*(lsquare[2,1]*l[a,1]*lcon[b,2]-lsquare[1,2]*l[a,2]*lcon[b,1])*(M[1,1]/M[2,2])*self.f1(M[2,2]/M[1,1])
        second      = 1j*(2./3.)*(1/(np.power(M[0,0]/M[1,1],2)-1.))*(l[a,1]*lcon[b,0]*lsquare[1,0]-lcon[b,1]*l[a,0]*lsquare[0,1])
        fourth      = 1j*(2./3.)*(1/(np.power(M[2,2]/M[1,1],2)-1.))*(l[a,1]*lcon[b,2]*lsquare[1,2]-lcon[b,1]*l[a,2]*lsquare[2,1])
        epsilon2abtemp = prefactor*(first+second+third+fourth)
        return epsilon2abtemp
        #################################

    @property
    def isPerturbative(self):
        """
        Check perturbativity of Yukawas
        """
        y = self.h
        #limit of perturbativity for y
        limit = np.power(4*np.pi,0.5)
        #check if any element of column 1, 2 or 3 is larger than limit
        col1               = (y[0,0] < limit)*(y[1,0] < limit)*(y[2,0] < limit)
        col2               = (y[0,1] < limit)*(y[1,1] < limit)*(y[2,1] < limit)
        col3               = (y[0,2] < limit)*(y[1,2] < limit)*(y[2,2] < limit)
        return col1*col2*col3

    #################################
    #Check we are not in resonance  #
    #################################

    def resonance(self, z):
        """
        calculate decay rate Gamma in terms of the total epsilon (epstot)
        """
        eps1tau = np.real(self.epsilon(0,1,2,2))
        eps1mu  = np.real(self.epsilon(0,1,2,1))
        eps1e   = np.real(self.epsilon(0,1,2,0))
        epstot  = eps1tau+eps1mu+eps1e
        k       = self.k1
        d       = self.D1(k,z)
        Gamma   = (self.M1**2/self.MP)*np.sqrt(2*np.pi/3)*np.sqrt((np.pi**2)*self.gstar/30)*(1+epstot)*d/z

        #calculate (decay rate)/(mass splitting)
        return Gamma/(M2-M1)


    ################################################
    #RHS of ODE for derivative of N1, Ntau, Nmu, Ne#
    ################################################
    def RHS_1DS_DM(self, y0,z,epstt,epsmm,epsee,epstm,epste,epsme,c1t,c1m,c1e,k):
        N1      = y0[0]
        Ntt     = y0[1]
        Nmm     = y0[2]
        Nee     = y0[3]
        Ntm     = y0[4]
        Nte     = y0[5]
        Nme     = y0[6]

        d       = np.real(self.D1(k,z))
        w1      = np.real(self.W1(k,z))
        n1eq    = self.N1Eq(z)

        c1tc    = np.conjugate(c1t)
        c1mc    = np.conjugate(c1m)
        c1ec    = np.conjugate(c1e)

        widtht  = 485e-10*self.MP/self.M1
        widthm  = 1.7e-10*self.MP/self.M1


        #define the different RHSs for each equation
        rhs1 =      -d*(N1-n1eq)

        rhs2 = epstt*d*(N1-n1eq)-0.5*w1*(2*c1t*c1tc*Ntt + c1m*c1tc*Ntm + c1e*c1tc*Nte + np.conjugate(c1m*c1tc*Ntm+c1e*c1tc*Nte)                  )
        rhs3 = epsmm*d*(N1-n1eq)-0.5*w1*(2*c1m*c1mc*Nmm + c1m*c1tc*Ntm + c1e*c1mc*Nme + np.conjugate(c1m*c1tc*Ntm+c1e*c1mc*Nme)                  )
        rhs4 = epsee*d*(N1-n1eq)-0.5*w1*(2*c1e*c1ec*Nee + c1e*c1mc*Nme + c1e*c1tc*Nte + np.conjugate(c1e*c1mc*Nme+c1e*c1tc*Nte)                  )
        rhs5 = epstm*d*(N1-n1eq)-0.5*w1*(  c1t*c1mc*Nmm + c1e*c1mc*Nte + c1m*c1mc*Ntm + c1mc*c1t*Ntt + c1t*c1tc*Ntm + c1t*c1ec*np.conjugate(Nme) ) - widtht*Ntm - widthm*Ntm
        rhs6 = epste*d*(N1-n1eq)-0.5*w1*(  c1t*c1ec*Nee + c1e*c1ec*Nte + c1m*c1ec*Ntm + c1t*c1ec*Ntt + c1t*c1mc*Nme + c1t*c1tc*Nte               ) - widtht*Nte
        rhs7 = epsme*d*(N1-n1eq)-0.5*w1*(  c1m*c1ec*Nee + c1e*c1ec*Nme + c1m*c1ec*Nmm + c1t*c1ec*np.conjugate(Ntm)  + c1m*c1mc*Nme + c1m*c1tc*Nte) - widthm*Nme

        return [rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7]

    def RHS_1DS_Approx(self, y0,z,epstt,epsmm,epsee,c1t,c1m,c1e,k):
        N1      = y0[0]
        Ntt     = y0[1]
        Nmm     = y0[2]
        Nee     = y0[3]

        d       = np.real(self.D1(k,z))
        w1      = np.real(self.W1(k,z))
        n1eq    = self.N1Eq(z)

        c1tc    = np.conjugate(c1t)
        c1mc    = np.conjugate(c1m)
        c1ec    = np.conjugate(c1e)


        #define the different RHSs for each equation
        rhs1 =      -d*(N1-n1eq)

        rhs2 = epstt*d*(N1-n1eq)-0.5*w1*(2*c1t*c1tc*Ntt)
        rhs3 = epsmm*d*(N1-n1eq)-0.5*w1*(2*c1m*c1mc*Nmm)
        rhs4 = epsee*d*(N1-n1eq)-0.5*w1*(2*c1e*c1ec*Nee)

        return [rhs1, rhs2, rhs3, rhs4]

    def RHS_2DS_Approx(self, y0, z, ETA, C, K):
        N1, N2, Ntt, Nmm, Nee = y0
        eps1tt,eps1mm,eps1ee,eps1tm,eps1te,eps1me,eps2tt,eps2mm,eps2ee,eps2tm,eps2te,eps2me = ETA
        c1t,c1m,c1e,c2t,c2m,c2e = C
        k1term,k2term = K

        d1      = np.real(self.D1(k1term, z))
        w1      = np.real(self.W1(k1term, z))
        d2      = np.real(self.D2(k2term, z))
        w2      = np.real(self.W2(k2term, z))
        n1eq    = self.N1Eq(z)
        n2eq    = self.N2Eq(z)

        c1tc    = np.conjugate(c1t)
        c1mc    = np.conjugate(c1m)
        c1ec    = np.conjugate(c1e)

        c2tc    = np.conjugate(c2t)
        c2mc    = np.conjugate(c2m)
        c2ec    = np.conjugate(c2e)

        #define the different RHSs for each equation
        rhs1 =      -d1*(N1-n1eq)
        rhs2 =      -d2*(N2-n2eq)
        rhs3 = eps1tt*d1*(N1-n1eq)+eps2tt*d2*(N2-n2eq)-0.5*w1*(2*c1t*c1tc*Ntt) -0.5*w2*(2*c2t*c2tc*Ntt)

        rhs4 = eps1mm*d1*(N1-n1eq)+eps2mm*d2*(N2-n2eq)-0.5*w1*(2*c1m*c1mc*Nmm) -0.5*w2*(2*c2m*c2mc*Nmm)

        rhs5 = eps1ee*d1*(N1-n1eq)+eps2ee*d2*(N2-n2eq)-0.5*w1*(2*c1e*c1ec*Nee) -0.5*w2*(2*c2e*c2ec*Nee)

        RHStemp = [rhs1, rhs2, rhs3, rhs4, rhs5]
        return RHStemp

    def RHS_2DS_DM(self, y0, zzz, ETA, C, K, W):
        N1, N2, Ntt, Nmm, Nee, Ntm, Nte, Nme = y0
        eps1tt,eps1mm,eps1ee,eps1tm,eps1te,eps1me,eps2tt,eps2mm,eps2ee,eps2tm,eps2te,eps2me = ETA
        c1t,c1m,c1e,c2t,c2m,c2e = C
        k1term,k2term = K
        widtht,widthm = W
        d1      = np.real(self.D1(k1term, zzz))
        w1      = np.real(self.W1(k1term, zzz))
        d2      = np.real(self.D2(k2term, zzz))
        w2      = np.real(self.W2(k2term, zzz))
        n2eq    = self.N2Eq(zzz)
        n1eq    = self.N1Eq(zzz)


        c1tc    = np.conjugate(c1t)
        c1mc    = np.conjugate(c1m)
        c1ec    = np.conjugate(c1e)

        c2tc    = np.conjugate(c2t)
        c2mc    = np.conjugate(c2m)
        c2ec    = np.conjugate(c2e)

        #define the different RHSs for each equation
        rhs1 =      -d1*(N1-n1eq)
        rhs2 =      -d2*(N2-n2eq)
        rhs3 = eps1tt*d1*(N1-n1eq)+eps2tt*d2*(N2-n2eq)-0.5*w1*(2*c1t*c1tc*Ntt+c1m*c1tc*Ntm+c1e*c1tc*Nte+np.conjugate(c1m*c1tc*Ntm+c1e*c1tc*Nte))-0.5*w2*(2*c2t*c2tc*Ntt+c2m*c2tc*Ntm+c2e*c2tc*Nte+np.conjugate(c2m*c2tc*Ntm+c2e*c2tc*Nte))

        rhs4 = eps1mm*d1*(N1-n1eq)+eps2mm*d2*(N2-n2eq)-0.5*w1*(2*c1m*c1mc*Nmm+c1m*c1tc*Ntm+c1e*c1mc*Nme+np.conjugate(c1m*c1tc*Ntm+c1e*c1mc*Nme))-0.5*w2*(2*c2m*c2mc*Nmm+c2m*c2tc*Ntm+c2e*c2mc*Nme+np.conjugate(c2m*c2tc*Ntm+c2e*c2mc*Nme))

        rhs5 = eps1ee*d1*(N1-n1eq)+eps2ee*d2*(N2-n2eq)-0.5*w1*(2*c1e*c1ec*Nee+c1e*c1mc*Nme+c1e*c1tc*Nte+np.conjugate(c1e*c1mc*Nme+c1e*c1tc*Nte))-0.5*w2*(2*c2e*c2ec*Nee+c2e*c2mc*Nme+c2e*c2tc*Nte+np.conjugate(c2e*c2mc*Nme+c2e*c2tc*Nte))

        rhs6 = eps1tm*d1*(N1-n1eq)+eps2tm*d2*(N2-n2eq)-0.5*w1*(c1t*c1mc*Nmm+c1e*c1mc*Nte+c1m*c1mc*Ntm+c1mc*c1t*Ntt+c1t*c1tc*Ntm+c1t*c1ec*np.conjugate(Nme))-0.5*w2*(c2t*c2mc*Nmm+c2e*c2mc*Nte+c2m*c2mc*Ntm+c2mc*c2t*Ntt+c2t*c2tc*Ntm+c2t*c2ec*np.conjugate(Nme))-widtht*Ntm-widthm*Ntm

        rhs7 = eps1te*d1*(N1-n1eq)+eps2te*d2*(N2-n2eq)-0.5*w1*(c1t*c1ec*Nee+c1e*c1ec*Nte+c1m*c1ec*Ntm+c1t*c1ec*Ntt+c1t*c1mc*Nme+c1t*c1tc*Nte)-0.5*w2*(c2t*c2ec*Nee+c2e*c2ec*Nte+c2m*c2ec*Ntm+c2t*c2ec*Ntt+c2t*c2mc*Nme+c2t*c2tc*Nte)-widtht*Nte

        rhs8 = eps1me*d1*(N1-n1eq)+eps2me*d2*(N2-n2eq)-0.5*w1*(c1m*c1ec*Nee+c1e*c1ec*Nme+c1m*c1ec*Nmm+c1t*c1ec*np.conjugate(Ntm)+c1m*c1mc*Nme+c1m*c1tc*Nte)-0.5*w2*(c2m*c2ec*Nee+c2e*c2ec*Nme+c2m*c2ec*Nmm+c2t*c2ec*np.conjugate(Ntm)+c2m*c2mc*Nme+c2m*c2tc*Nte)-widthm*Nme

        RHStemp = [rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8]
        return RHStemp

    @property
    def getEtaB_2DS_Approx(self):
        #Define fixed quantities for BEs  
        _ETA = [
            np.real(self.epsilon(0,1,2,2)),
            np.real(self.epsilon(0,1,2,1)),
            np.real(self.epsilon(0,1,2,0)),
            np.real(self.epsilon(1,0,2,2)),
            np.real(self.epsilon(1,0,2,1)),
            np.real(self.epsilon(1,0,2,0))
            ]

        _HT = [
            np.real(self.hterm(2,0)),
            np.real(self.hterm(1,0)),
            np.real(self.hterm(0,0)),
            np.real(self.hterm(2,1)),
            np.real(self.hterm(1,1)),
            np.real(self.hterm(0,1))
            ]

        _K      = [np.real(self.k1), np.real(self.k2)]
        y0      = np.array([0+0j,0+0j,0+0j,0+0j,0+0j], dtype=np.complex128)

        # KKK
        _ETA = [
            np.real(self.epsilon1ab(2,2)),
            np.real(self.epsilon1ab(1,1)),
            np.real(self.epsilon1ab(0,0)),
                    self.epsilon1ab(2,1) ,
                    self.epsilon1ab(2,0) ,
                    self.epsilon1ab(1,0),
            np.real(self.epsilon2ab(2,2)),
            np.real(self.epsilon2ab(1,1)),
            np.real(self.epsilon2ab(0,0)),
                    self.epsilon2ab(2,1) ,
                    self.epsilon2ab(2,0) ,
                    self.epsilon2ab(1,0),
            ]
        _C = [  self.c1a(2), self.c1a(1), self.c1a(0),
                self.c2a(2), self.c2a(1), self.c2a(0)]
        _K = [np.real(self.k1), np.real(self.k2)]

        ys      = odeintw(self.RHS_2DS_Approx, y0, self.xs, args = tuple([_ETA, _C, _K]))
        nb      = 0.013*(ys[-1,2]+ys[-1,3]+ys[-1,4])

        return nb

    @property
    def getEtaB_2DS_DM(self):

        #Define fixed quantities for BEs
        _ETA = [
            np.real(self.epsilon1ab(2,2)),
            np.real(self.epsilon1ab(1,1)),
            np.real(self.epsilon1ab(0,0)),
                    self.epsilon1ab(2,1) ,
                    self.epsilon1ab(2,0) ,
                    self.epsilon1ab(1,0),
            np.real(self.epsilon2ab(2,2)),
            np.real(self.epsilon2ab(1,1)),
            np.real(self.epsilon2ab(0,0)),
                    self.epsilon2ab(2,1) ,
                    self.epsilon2ab(2,0) ,
                    self.epsilon2ab(1,0),
            ]
        _C = [  self.c1a(2), self.c1a(1), self.c1a(0),
                self.c2a(2), self.c2a(1), self.c2a(0)]
        _K = [np.real(self.k1), np.real(self.k2)]
        _W = [ 485e-10*self.MP/self.M1, 1.7e-10*self.MP/self.M1]

        y0      = np.array([0+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j], dtype=np.complex128)

        zcrit   = 1e100
        ys, _      = odeintw(self.RHS_2DS_DM, y0, self.xs, args = tuple([_ETA, _C , _K, _W]), full_output=1)
        nb      = np.real(0.013*(ys[-1,2]+ys[-1,3]+ys[-1,4]))

        X=_["tcur"]
        N1     =ys[:,0][0:-1]
        N2     =ys[:,1][0:-1]
        Nee    =ys[:,2][0:-1]
        Nmumu  =ys[:,3][0:-1]
        Ntautau=ys[:,4][0:-1]


        import pylab

        pylab.plot(X, np.absolute(Nee), label="|Nee|"        , color="r")
        pylab.plot(X, np.absolute(Nmumu), label="|Nmumu|"    , color="g")
        pylab.plot(X, np.absolute(Ntautau), label="|Ntautau|", color="b")

        pylab.legend()
        pylab.ylabel("|N|")
        pylab.xlabel("z")
        pylab.xscale("log")
        pylab.yscale("log")
        pylab.xlim((self.xmin,self.xmax))

        pylab.show()



        pylab.plot(X, np.absolute(Nee+Nmumu+Ntautau), label="2 dec. st. (DM) eta=%e"%nb, color="b")
        pylab.legend()
        pylab.ylabel("eta")
        pylab.xlabel("z")
        pylab.xscale("log")
        pylab.yscale("log")
        pylab.xlim((self.xmin,self.xmax))

        pylab.show()


        return nb

    def writeTXT(self, X, Y, fname):
        Z=np.array([x for  x in zip(list(X),list(Y))])
        np.savetxt(self.plotprefix+fname, Z)



    @property
    def getEtaB_1DS_DM(self):
        #Define fixed quantities for BEs   
        epstt = np.real(self.epsilonab(2,2))
        epsmm = np.real(self.epsilonab(1,1))
        epsee = np.real(self.epsilonab(0,0))
        epstm =         self.epsilonab(2,1)
        epste =         self.epsilonab(2,0)
        epsme =         self.epsilonab(1,0)

        c1t   =                 self.c1a(2)
        c1m   =                 self.c1a(1)
        c1e   =                 self.c1a(0)

        xs      = np.linspace(self.xmin, self.xmax, self.xsteps)
        k       = np.real(self.k1)
        y0      = np.array([0+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j], dtype=np.complex128)

        params  = np.array([epstt,epsmm,epsee,epstm,epste,epsme,c1t,c1m,c1e,k], dtype=np.complex128)

        ys, _      = odeintw(self.RHS_1DS_DM, y0, self.xs, args = tuple(params), full_output=1)
        nb      = 0.013*(ys[-1,1]+ys[-1,2]+ys[-1,3]).real

        X=_["tcur"]
        # ys=np.absolute(ys)
        N1     =ys[:,0][0:-1]
        Nee    =ys[:,1][0:-1]
        Nmumu  =ys[:,2][0:-1]
        Ntautau=ys[:,3][0:-1]


        import pylab
        fig, ax = pylab.subplots()

        pylab.plot(X, np.absolute(Nee), label="|Nee|"        , color="r")
        pylab.plot(X, np.absolute(Nmumu), label="|Nmumu|"    , color="g")
        pylab.plot(X, np.absolute(Ntautau), label="|Ntautau|", color="b")

        pylab.legend()
        pylab.xlabel("z")
        pylab.ylabel("|N|")
        pylab.xscale("log")
        pylab.yscale("log")
        pylab.xlim((self.xmin,self.xmax))
        pylab.show()


        pylab.clf()
        pylab.plot(X.real, Nee.real+Nmumu.real+Ntautau.real, label="1 dec. st. (DM) eta=%e"%nb, color="b")
        pylab.legend()
        pylab.xlabel("z")
        pylab.ylabel("eta")
        pylab.xscale("log")
        pylab.yscale("log")
        pylab.xlim((self.xmin,self.xmax))
        pylab.show()

        return nb

    @property
    def getEtaB_1DS_Approx(self):
        #Define fixed quantities for BEs   
        epstt = np.real(self.epsilonab(2,2))
        epsmm = np.real(self.epsilonab(1,1))
        epsee = np.real(self.epsilonab(0,0))

        c1t   =                 self.c1a(2)
        c1m   =                 self.c1a(1)
        c1e   =                 self.c1a(0)

        xs      = np.linspace(self.xmin, self.xmax, self.xsteps)
        k       = np.real(self.k1)
        y0      = np.array([0+0j,0+0j,0+0j,0+0j], dtype=np.complex128)

        params  = np.array([epstt,epsmm,epsee,c1t,c1m,c1e,k], dtype=np.complex128)

        ys      = odeintw(self.RHS_1DS_Approx, y0, self.xs, args = tuple(params))
        nb      = 0.013*(ys[-1,1]+ys[-1,2]+ys[-1,3])

        return nb

    @property
    def EtaB(self):
        if self.nds==1:
            if self.approx:
                return np.real(self.getEtaB_1DS_Approx)
            else:
                return np.real(self.getEtaB_1DS_DM)
        if self.nds==2:
            if self.approx:
                return np.real(self.getEtaB_2DS_Approx)
            else:
                return np.real(self.getEtaB_2DS_DM)

