import numpy as np
from numpy.polynomial import hermite, legendre
from scipy import linalg


C = 1/np.sqrt(2*np.pi)
def normal_pdf(x, sigma):
    return C/sigma*np.exp(-(x/sigma)**2/2)


"""General code for Gauss-Hermite and Gauss-Legendre quadrature
If needed, could speed up hermite_std and legendre_interval using Numba"""

def hermite_std(S, sigma):
    """Given raw S output by hermite.hermgauss, scale weights and nodes
    so that they integrate against a normal with standard deviation sigma"""
    z, wnorm = S
    x = np.sqrt(2)*sigma*z
    w = wnorm / np.sqrt(np.pi)
    return w, x

def hermite_quick(n, sigma):
    """Convenience function that returns Gauss-Hermite quadrature weights
    and nodes that integrate against a normal with standard deviation sigma.
    Avoid this and precompute hermgauss if using too many times."""
    return hermite_std(hermite.hermgauss(n), sigma)

def legendre_interval(S, a, b):
    """Given raw S output by legendre.leggauss, map weights and nodes
    so that they integrate over an interval [a,b]"""
    z, wnorm = S
    x = _demap(z, a, b)
    w = (b-a)/2*wnorm
    return w, x

def legendre_quick(n, a, b):
    return legendre_interval(legendre.leggauss(n), a, b)

def _demap(z, a, b):
    """Map z in [-1,1] to x in [a,b]"""
    return (b-a)/2*(z+1) + a


"""Applied code for integration using quadrature that we use"""

# precompute default Gauss-Hermite and Gauss-Legendre raw weights/nodes
# note that we need many more for Gauss-Legendre, since we'll use this to
# integrate trickier functions
# can modify this as desired
Herm = hermite.hermgauss(7)
Leg = legendre.leggauss(50)


def expectations_normal(f, xs, sigma):
    """Take expectations f(x+sigma*eps) for
    std normal eps at each of a vector of xs"""
    wh, epsh = hermite_std(Herm, sigma)
    xps = epsh[:, np.newaxis] + xs
    fxps = f(xps.ravel()).reshape(xps.shape)
    return wh @ fxps

def integrate(f, a, b):
    """Integrate f on [a, b]"""
    wl, xl = legendre_interval(Leg, a, b)
    return wl @ f(xl)


"""Code for dealing with Jacobians and time-dependent pricers"""

def J_from_F(F, T):
    """From fake news matrix F (possibly smaller than T*T), build Jacobian J of size T*T,
    using recursion J_(t,s) = F_(t,s) + J_(t-1,s-1), base case J_(t,s) = F(t,s) for t or s=0,
    and imputing F_(t,s)=0 when (t,s) outside dimensions of F"""
    if T < len(F):
        raise ValueError(f"T={T} must be weakly larger than existing size of F, {len(F)}")

    J = np.zeros((T, T))
    J[:len(F), :len(F)] = F
    for t in range(1, J.shape[1]):
        J[1:, t] += J[:-1, t - 1]
    return J


def F_for_td(Phi, beta):
    """Build fake news matrix F for Psi for a time-dependent pricer
     with survival Phi and discount rate beta"""
    columns = beta**np.arange(len(Phi))*Phi
    columns /= columns.sum()

    rows = Phi / Phi.sum()
    return np.outer(rows, columns)


def Psi_for_td(Phi, beta, T):
    """Build T-by-T Psi for time-dependent pricer
    with survival curve Phi and discount rate beta"""
    return J_from_F(F_for_td(Phi, beta), T)
