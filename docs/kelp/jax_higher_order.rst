Higher-order harmonics in the JAX backend
=========================================

The JAX backend (``kelp.jax``) supports the full :math:`h_{m\ell}` basis up to
arbitrary :math:`\ell_{max}`, not just the :math:`m=\ell=1` term. This makes it
possible to fit or sample higher-order structure with gradient-based methods
(e.g. numpyro).

.. note::

    **API change.** The fifth argument of
    ``kelp.jax.thermal_phase_curve`` was renamed from ``C_11`` to ``C_ml``.
    Positional callers are unaffected, and a scalar ``C_ml`` behaves exactly as
    ``C_11`` did (it is the power in :math:`m=\ell=1`). Callers passing the
    keyword ``C_11=...`` must switch to ``C_ml=...``.

The ``C_ml`` format
-------------------

``C_ml`` is either a scalar, or a ragged nested list in the same format as
`~kelp.Model`, indexed ``C_ml[l][m]``. Row ``l`` must have :math:`2\ell+1`
entries, and negative :math:`m` follows Python negative-index wraparound, so
row ``l`` is

.. code-block:: python

    [C_l0, C_l1, ..., C_ll, C_l-l, ..., C_l-1]

For example, with :math:`\ell_{max}=2`:

.. code-block:: python

    C_ml = [[0],
            [0, C_11, 0],
            [0, C_21, C_22, C_2m2, C_2m1]]

and ``lmax = len(C_ml) - 1``. The :math:`m=0` entries do not contribute.
The number of rows is static when the function is traced, but the entries may
be JAX tracers, so ``jax.jit`` and ``jax.grad`` work with respect to any entry.
Dense 2D arrays are rejected, because the negative-index wraparound would mean
something different.

Temperature map and phase curve for :math:`\ell_{max}=1` and 2
--------------------------------------------------------------

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u

    from kelp import Filter, Planet
    from kelp.jax import thermal_phase_curve

    hotspot_offset = -0.8
    alpha, omega_drag, f = 0.6, 4.5, 2**-0.5
    p = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')
    filt.bin_down(10)

    C_lmax1 = [[0], [0, 0.18, 0]]
    C_lmax2 = [[0], [0, 0.18, 0], [0, 0.03, 0.04, -0.02, 0.01]]

    phi = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 50)
    theta2d, phi2d = np.meshgrid(theta, phi)
    xi = np.linspace(-np.pi, np.pi, 100)

    def run(C):
        return thermal_phase_curve(
            xi, hotspot_offset, omega_drag, alpha, C, p.T_s, p.a, p.rp_a, 0,
            theta2d, phi2d, filt.wavelength.to(u.m).value,
            filt.transmittance, f)

    pc1, T1 = run(C_lmax1)
    pc2, T2 = run(C_lmax2)
    # T has shape (n_phi, n_theta, n_xi, 1); the map does not depend on xi
    T1, T2 = np.asarray(T1)[:, :, 0, 0].T, np.asarray(T2)[:, :, 0, 0].T

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)
    vmin, vmax = min(T1.min(), T2.min()), max(T1.max(), T2.max())
    for i, (T, label) in enumerate([(T1, 'lmax=1'), (T2, 'lmax=2')]):
        ax = fig.add_subplot(gs[0, i], projection='mollweide')
        cax = ax.pcolormesh(phi, theta - np.pi / 2, T, vmin=vmin, vmax=vmax)
        plt.colorbar(cax, ax=ax, label='T [K]', orientation='horizontal')
        ax.set(title=label, xlabel='longitude [deg]', ylabel='latitude [deg]')

    dT = T2 - T1
    vm = np.abs(dT).max()
    ax = fig.add_subplot(gs[0, 2], projection='mollweide')
    cax = ax.pcolormesh(phi, theta - np.pi / 2, dT, cmap='RdBu_r',
                        vmin=-vm, vmax=vm)
    plt.colorbar(cax, ax=ax, label='T(lmax=2) - T(lmax=1) [K]',
                 orientation='horizontal')
    ax.set(title='difference (not z-scores)', xlabel='longitude [deg]',
           ylabel='latitude [deg]')

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(xi, 1e6 * np.asarray(pc1), label='lmax=1')
    ax.plot(xi, 1e6 * np.asarray(pc2), label='lmax=2')
    ax.set(xlabel='orbital phase angle $\\xi$ [rad]', ylabel='$F_p/F_s$ [ppm]')
    ax.legend()
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(xi, 1e6 * (np.asarray(pc2) - np.asarray(pc1)), color='k')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set(xlabel='orbital phase angle $\\xi$ [rad]',
           ylabel='lmax=2 - lmax=1 [ppm]')
    fig.tight_layout()
    plt.show()

JAX versus Cython
-----------------

The JAX and Cython implementations agree to within float32 precision for the
temperature map, and to better than 1% for the phase curve. Signed residuals
are shown (these are not z-scores).

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u

    from kelp import Filter, Planet, Model
    from kelp.jax import thermal_phase_curve

    hotspot_offset, alpha, omega_drag, f = -0.8, 0.6, 4.5, 2**-0.5
    p = Planet.from_name('HD 189733')
    filt = Filter.from_name('IRAC 1')
    filt.bin_down(10)
    lmax = 3
    rng = np.random.default_rng(1)
    C_ml = [[0]] + [list(rng.uniform(-0.05, 0.05, 2 * l + 1))
                    for l in range(1, lmax + 1)]
    C_ml[1][1] = 0.18

    model = Model(hotspot_offset, alpha, omega_drag, 0, C_ml, lmax,
                  p.a, p.rp_a, p.T_s, filt=filt)
    n_theta, n_phi = 50, 100
    T_cy, theta, phi = model.temperature_map(n_theta, n_phi, f=f)

    theta2d, phi2d = np.meshgrid(theta, phi)
    xi = np.linspace(-np.pi, np.pi, 100)
    pc_jax, T_jax = thermal_phase_curve(
        xi, hotspot_offset, omega_drag, alpha, C_ml, p.T_s, p.a, p.rp_a, 0,
        theta2d, phi2d, filt.wavelength.to(u.m).value, filt.transmittance, f)
    T_jax = np.asarray(T_jax)[:, :, 0, 0].T
    pc_cy = 1e-6 * model.thermal_phase_curve(xi, f=f).flux

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
    for i, (T, label) in enumerate([(T_cy, 'Cython'), (T_jax, 'JAX')]):
        ax = fig.add_subplot(gs[0, i], projection='mollweide')
        cax = ax.pcolormesh(phi, theta - np.pi / 2, T)
        plt.colorbar(cax, ax=ax, label='T [K]', orientation='horizontal')
        ax.set(title=label, xlabel='longitude [deg]', ylabel='latitude [deg]')
    res = T_jax - T_cy
    vm = max(np.abs(res).max(), 1e-12)
    ax = fig.add_subplot(gs[0, 2], projection='mollweide')
    cax = ax.pcolormesh(phi, theta - np.pi / 2, res, cmap='RdBu_r',
                        vmin=-vm, vmax=vm)
    plt.colorbar(cax, ax=ax, label='JAX - Cython [K], not z-scores',
                 orientation='horizontal')
    ax.set(title='residual', xlabel='longitude [deg]',
           ylabel='latitude [deg]')

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(xi, 1e6 * pc_cy, label='Cython')
    ax.plot(xi, 1e6 * np.asarray(pc_jax), '--', label='JAX')
    ax.set(xlabel='orbital phase angle $\\xi$ [rad]', ylabel='$F_p/F_s$ [ppm]')
    ax.legend()
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(xi, 1e6 * (np.asarray(pc_jax) - pc_cy), color='k')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set(xlabel='orbital phase angle $\\xi$ [rad]',
           ylabel='JAX - Cython [ppm]')
    fig.tight_layout()
    plt.show()
