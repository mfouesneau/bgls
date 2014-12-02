Bayesian Generalized Lomb-Scargle periodogram
=============================================

The Lomb-Scargle periodogram was developed by Lomb [Lomb76]_ and further extended
by Scargle [Scargle82]_ to find, and test the significance of weak periodic signals
with uneven temporal sampling.

This repository implements the method described in Mortier et al. (2014)
[Mortier14]_, that is a Bayesian formalism for the generalized Lomb-Scargle
periodogram.


Requires numpy (http://www.numpy.org/)

Tested on Python 2.7, and Python 3


Reference
---------


.. [Lomb76] N.R. Lomb "Least-squares frequency analysis of unequally spaced
            data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976

.. [Scargle82] J.D. Scargle "Studies in astronomical time series analysis. II - 
               Statistical aspects of spectral analysis of unevenly spaced data",
               The Astrophysical Journal, vol 263, pp. 835-853, 1982

.. [Mortier14] Mortier et al., 2014, http://arxiv.org/abs/1412.0467



Example
-------

.. code:: python

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
