import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, ifftshift, fftfreq
from scipy.signal.windows import tukey


# ChatGPT o1 generated:
# TODO: check line-by-line


def wigner(
    signal: np.ndarray,
    zero_pad_factor: float = 2.0,
    window: str or callable or None = None,
    dx: float = 1.0,
    apply_half_sample: bool = True,
    do_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 1D Wigner distribution (Wigner–Ville transform) of a discrete signal,
    with optional zero-padding, windowing, half-sample interpolation, and normalization.

    The continuous-time Wigner transform of a signal s(t) is defined as:

    .. math::

        W(t, f) = \\int_{-\\infty}^{\\infty}
        s\\bigl(t + \\tfrac{\\tau}{2}\\bigr) \\,
        s^*\\bigl(t - \\tfrac{\\tau}{2}\\bigr) \\,
        e^{- i \\, 2\\pi \\, f \\tau}
        \\; d\\tau,

    which in discrete form typically requires evaluations at half-sample offsets.
    This function applies interpolation as needed, and also provides windowing
    and zero-padding to reduce wraparound or boundary artifacts.

    Parameters
    ----------
    signal : np.ndarray of shape (N,)
        One-dimensional signal array. Can be real or complex.
    zero_pad_factor : float, optional
        Factor by which to increase the signal length via zero-padding.
        The padded length will be int(N * zero_pad_factor).
        Defaults to 2.0 (2× padding).
    window : {'hann', 'tukey'} or callable or None, optional
        Window function applied to the zero-padded signal to mitigate edge effects.
        - If 'hann', a Hann window is used.
        - If 'tukey', a Tukey window with alpha=0.25 is used.
        - If a callable, it should accept an integer `M` and return a 1D window of length `M`.
        - If None, no window is applied.
        Defaults to None.
    dx : float, optional
        Sample spacing in the time (or spatial) domain. Used to set up arrays
        for the returned `t_array` and frequency array. Default is 1.0.
    apply_half_sample : bool, optional
        If True, use half-sample interpolation to approximate s(t ± m/2). If False,
        use integer shifts s(t ± m). Default is True.
    do_normalize : bool, optional
        If True, multiply the resulting Wigner distribution by 1/N after FFT
        to approximate the continuous marginals. Default is True.

    Returns
    -------
    W : np.ndarray of shape (M, M)
        The computed Wigner distribution, indexed as W[t_index, f_index].
        Rows typically correspond to time, columns to frequency.
    t_array : np.ndarray of shape (M,)
        Discrete "time" axis corresponding to rows of W. Centered so that t=0
        is in the middle of the array (if zero_pad_factor > 1).
    f_array : np.ndarray of shape (M,)
        Discrete frequency axis corresponding to columns of W, shifted so that
        f=0 is in the middle (if the FFT shift is applied).

    Notes
    -----
    - The Wigner distribution is a quasi-probability distribution and can take
      negative values due to interference terms.
    - Interpolation is done via a linear spline. You may replace it with a more
      sophisticated method if desired.
    - Zero-padding and windowing help reduce wraparound (circular) artifacts in
      the correlation step.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> # Generate a simple two-tone signal
    >>> N = 256
    >>> t = np.arange(N)
    >>> signal = np.sin(2*np.pi*0.05*t) + 0.6*np.sin(2*np.pi*0.1*t)
    >>> # Compute Wigner distribution
    >>> W, t_axis, f_axis = wigner(
    ...     signal,
    ...     zero_pad_factor=2,
    ...     window='hann',
    ...     dx=1.0,
    ...     apply_half_sample=True,
    ...     do_normalize=True
    ... )
    >>> # Plot results
    >>> plt.figure(figsize=(8, 6))
    >>> extent = [f_axis[0], f_axis[-1], t_axis[0], t_axis[-1]]
    >>> plt.imshow(W, extent=extent, origin='lower', aspect='auto', cmap='jet')
    >>> plt.colorbar(label='Wigner amplitude')
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Time')
    >>> plt.title('Wigner Distribution')
    >>> plt.show()
    """
    N_orig = len(signal)
    N_padded = int(np.round(N_orig * zero_pad_factor))
    half_pad = (N_padded - N_orig) // 2

    # ------------------
    # 1) Zero-pad
    # ------------------
    sig_padded = np.zeros(N_padded, dtype=signal.dtype)
    sig_padded[half_pad : half_pad + N_orig] = signal

    # ------------------
    # 2) Apply window
    # ------------------
    if window == "hann":
        w = np.hanning(N_padded)
        sig_padded *= w
    elif window == "tukey":
        w = tukey(N_padded, alpha=0.25)
        sig_padded *= w
    elif callable(window):
        w = window(N_padded)  # user-supplied
        sig_padded *= w
    # else None => no window

    # We'll rename this for clarity
    s = sig_padded
    N = N_padded

    # Build the "time" array (centered). You can shift if desired.
    t_array = dx * (np.arange(N) - N // 2)

    # ------------------
    # 3) Grid for correlation
    # ------------------
    t_grid = np.arange(N).reshape(-1, 1)  # shape (N,1)
    m_grid = np.arange(-N // 2, N - N // 2).reshape(1, -1)  # shape (1,N)

    # For half-sample:
    if apply_half_sample:
        t_plus = t_grid + 0.5 * m_grid
        t_minus = t_grid - 0.5 * m_grid
    else:
        t_plus = t_grid + m_grid
        t_minus = t_grid - m_grid

    # ------------------
    # 4) Interpolation
    # ------------------
    # We'll use linear interpolation, zero outside domain
    f_interp = interp1d(
        np.arange(N), s, kind="linear", bounds_error=False, fill_value=0.0
    )
    s_plus = f_interp(t_plus)
    s_minus = np.conjugate(f_interp(t_minus))

    # Correlation
    C = s_plus * s_minus  # shape (N,N)

    # ------------------
    # 5) FFT in m dimension
    # ------------------
    # shift -> ifftshift -> fft -> fftshift
    C_ifft = ifftshift(C, axes=1)
    C_fft = fft(C_ifft, axis=1)
    C_fft = fftshift(C_fft, axes=1)

    # Real part is typically the Wigner distribution
    W = np.real(C_fft)

    # ------------------
    # 6) Frequency array
    # ------------------
    f_array = fftshift(fftfreq(N, dx))

    # ------------------
    # 7) Optional normalization
    # ------------------
    if do_normalize:
        # 1/N factor is common for discrete Wigner so marginals approximate |s(t)|^2
        W *= 1.0 / N

    return W, t_array, f_array
