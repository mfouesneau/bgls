"""
Bayesian Generalized Lomb-Scargle periodogram

a vectorial implementation of Mortier et al. 2014 (arXiv:1412.0467v1) "BGLS:
    A Bayesian formalism for the generalised Lomb-Scargle periodogram"
"""
import numpy as np
from math import pi


def bgls(t, y, err, plow=0.5, phigh=100, ofac=1):
    """
    Bayesian Generalized Lomb-Scargle periodogram

    a vectorial implementation of Mortier et al. 2014 (arXiv:1412.0467v1) "BGLS:
        A Bayesian formalism for the generalised Lomb-Scargle periodogram"

    .. note::

        there is an internal prior that is a uniform probability distribution
        over the frequencies

    Parameters
    ----------
    t: ndarray
        time samples

    y: ndarray
        signal values at t

    err: ndarray
        signal uncertainties at t

    plow: float
        lowest period to consider

    phigh: float
        highest period to consider

    ofac: float
        oversampling factor

    Returns
    -------
    periods: ndarray
        extimated periods

    probs: ndarray
        probabilities associated to the periods
    """
    f = np.linspace(1./float(phigh), 1./float(plow), int(100 * float(ofac)))
    err2 = np.array(err) ** 2

    # eq 8
    w = 1. / err2

    # eq 9
    W = w.sum()

    # eq 10
    bigY = (w * np.array(y)).sum()

    p = []
    constants = []
    exponents = []

    omega = 2 * pi * f
    omegat = omega[:, None] * t[None, :]
    theta_up = (w[None, :] * np.sin(2 * omegat)).sum(1)
    theta_bo = (w[None, :] * np.cos(2 * omegat)).sum(1)

    theta = 0.5 * np.arctan2( theta_up, theta_bo)
    x = omegat - theta[:, None]

    cosx = np.cos(x)
    sinx = np.sin(x)
    wcosx = w[None, :] * cosx
    wsinx = w[None, :] * sinx

    # eq 14
    C = wcosx.sum(1)
    # eq 15
    S = wsinx.sum(1)

    # Eq 12
    YChat = (y * wcosx).sum(1)
    # eq 13
    YShat = (y * wsinx).sum(1)
    # eq 16
    CChat = (wcosx * cosx).sum(1)
    # eq 17
    SShat = (wsinx * sinx).sum(1)

    # below implements eqs 24, 25, & 26
    # init variables
    K = np.zeros(CChat.shape, dtype=float)
    L = np.zeros(CChat.shape, dtype=float)
    M = np.zeros(CChat.shape, dtype=float)
    constants = np.zeros(f.shape, dtype=float)

    # case 1
    ind = (CChat != 0) & (SShat != 0)
    tmp = 1. / (CChat[ind] * SShat[ind])
    K[ind] = (C[ind] * C[ind] * SShat[ind] + S[ind] * S[ind] * CChat[ind] - W * CChat[ind] * SShat[ind]) * 0.5 * tmp
    L[ind] = (bigY * CChat[ind] * SShat[ind] - C[ind] * YChat[ind] * SShat[ind] - S[ind] * YShat[ind] * CChat[ind]) * tmp
    M[ind] = (YChat[ind] * YChat[ind] * SShat[ind] + YShat[ind] * YShat[ind] * CChat[ind]) * 0.5 * tmp
    constants[ind] = (np.sqrt(tmp[ind] / abs(K[ind])))

    # case 2
    ind = CChat == 0
    K[ind] = (S[ind] * S[ind] - W * SShat[ind]) / (2. * SShat[ind])
    L[ind] = (bigY * SShat[ind] - S[ind] * YShat[ind]) / (SShat[ind])
    M[ind] = (YShat[ind] * YShat[ind]) / (2. * SShat[ind])
    constants[ind]  = (1. / np.sqrt(SShat[ind] * abs(K[ind])))

    ind = SShat == 0
    K[ind] = (C[ind] * C[ind] - W * CChat[ind]) / (2. * CChat[ind])
    L[ind] = (bigY * CChat[ind] - C[ind] * YChat[ind]) / (CChat[ind])
    M[ind] = (YChat[ind] * YChat[ind]) / (2. * CChat[ind])
    constants[ind] = (1. / np.sqrt(CChat[ind] * abs(K[ind])))

    exponents = (M - L * L / (4. * K))

    logp = np.log10(constants) + (exponents * np.log10(np.exp(1.)))

    # normalize to take power 10
    logp -= logp.max()
    p = 10 ** logp

    # normalize probs
    p /= p.sum()

    return 1. / f, p


def testunit():
    """ Make a dummy test and simple a figure for visualization """
    import pylab as plt

    # make signal
    period = 60.
    nperiods = 3
    npoints = 1000
    nsamples = 100

    # lb pars
    ofac = 20
    plow = 1
    phigh = 200

    SNR = 2
    t = np.linspace(0, nperiods * period, npoints)
    y = np.sin(2 * pi / float(period) * t)

    # take samples
    s = np.sort(np.random.randint(0, len(t), nsamples))
    ts = t[s]
    ys = y[s]
    yerr = ys * np.random.normal(0, 1. / SNR, len(ys))
    ysnoise = ys + yerr

    plt.figure()
    periods, probs = bgls(ts, ysnoise, ys / SNR, plow=plow, phigh=phigh, ofac=ofac)

    ax = plt.subplot(211)
    # ax.plot(t, y, 'k-')
    ax.errorbar(ts, ysnoise, yerr=ys / SNR, color='b', linestyle='None')
    ax.plot(t, np.sin(2 * pi / float(periods[probs.argmax()]) * t), 'r-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')

    ind = probs > 0.1
    v = np.random.choice(periods[ind], 10, p=probs[ind] / sum(probs[ind]))
    ax.plot(t, np.sin(2 * pi / v[:, None] * t).T, 'g-', alpha=0.3)

    ax = plt.subplot(212)
    ax.plot(periods, probs, 'k-')
    ax.vlines([period], 0, 1, color='b')
    ax.set_xlabel('Periods')
    ax.set_ylabel('Probabilities')
