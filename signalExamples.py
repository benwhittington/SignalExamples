import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def vanilla_signal_e(t, *args):
    return np.exp(-2 * np.pi * 1j * t)


def lfm_signal_e(t, t_range, f0, k, flip=False):
    if t_range is None:
        return np.exp(-(f0 - k * t/2) * 2 * np.pi * 1j * t)
    elif not flip:
        t0, t1 = t_range[0], t_range[1]
        out = np.zeros(t.shape[0], dtype=np.complex128)

        for i, t_i in enumerate(t):
            if t0 <= t_i and t_i <= np.abs(t1):
                out[i] = np.exp(-(2 * np.pi * 1j * (t_i - t0) * (f0 + k * (t_i - t0) / 2)))

        return out

    elif flip:
        t0, t1 = t_range[0], t_range[1]
        out = np.zeros(t.shape[0], dtype=np.complex128)

        for i, t_i in enumerate(t):
            if t0 <= t_i and t_i <= np.abs(t1):
                # out[i] = np.exp(-(2 * np.pi * 1j * (f0 + k * t1) * (t_i - t1) + np.pi * 1j * -k * (t_i - t1)**2))
                out[i] = np.exp(-(2 * np.pi * 1j * (t_i - t0) * (f0 + (t1 - t0) * k - k * (t_i - t0) / 2)))

        return out


def get_real_iq(x):
    return np.abs(x) * np.cos(np.angle(x))


def lfm_phi(t, t_range, f0, k):
    t0, t1 = t_range
    out = np.zeros(t.shape[0], dtype=np.complex128)

    t_out = np.array([t_i for t_i in t if (t0 <= t_i and t_i <= t1)])

    out = f0 + k * (t_out - t0) / 2

    return t_out, out


def lfm_demo0():
    f0 = 1  # baseband freq
    k = 1  # chrip-rate
    n_pts = 1.0e3
    t_min = 0
    t_max = 4 * np.pi

    t = np.linspace(t_min, t_max, int(n_pts))

    range_0 = [2, 6]
    range_1 = [5, 9]

    signal_0 = lfm_signal_e(t, range_0, f0, k, False)
    signal_1 = lfm_signal_e(t, range_1, f0, k, False)

    _, ax = plt.subplots(2, figsize=(10, 20))

    for a in ax:
        a.set_xticks([], []) 
        a.set_yticks([], []) 
        a.set_yticklabels([])
        a.set_xticklabels([])

    ax[0].plot(t, signal_0)
    ax[0].plot(t, signal_1)
    ax[0].set_ylim(-3, 3)
    ax[0].set_title("Time Domain LFM Data")

    ax[1].plot(*lfm_phi(t, range_0, f0, k))
    ax[1].plot(*lfm_phi(t, range_1, f0, k))
    ax[1].set_xlim(t_min, t_max)
    ax[1].set_title("Freq Domain LFM Data")

    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time")

    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Time")

    plt.show()


def lfm_demo1():
    f0 = 0.1  # baseband freq
    k = 0.2  # chrip-rate
    n_pts = 1.0e3
    t_min = 0
    t_max = 4 * np.pi

    t = np.linspace(t_min, t_max, int(n_pts))

    range_0 = [2, 10]

    signal_0 = lfm_signal_e(t, range_0, f0, k, False)

    _, ax = plt.subplots(2, figsize=(10, 20))

    ax[0].plot(t, signal_0)
    ax[0].set_title("Time Domain LFM Signal")
    
    ax[1].plot(*lfm_phi(t, range_0, f0, k))
    ax[1].set_xlim(t_min, t_max)
    ax[1].set_title("Frequency Domain LFM Signal")

    for a in ax:
        a.set_xticks([], []) 
        a.set_yticks([], []) 
        a.set_yticklabels([])
        a.set_xticklabels([])

    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time")

    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Time")

    plt.show()


def lfm_demo2():
    f0 = 0  # baseband freq
    k = 20  # chrip-rate
    n_pts = 500

    noise = np.random.normal(0, 0.5, n_pts)  # mean, sigma n_pts
    # noise = 0.5*np.random.randn(n_pts)

    t = np.linspace(0, 2 * np.pi, n_pts)

    signal_rx0 = lfm_signal_e(t, [1, 2], f0, k, True)  # signals from different reflections

    # signal_tx = lfm_signal_e(t, [0, 1], )


def main():

    f0 = 0  # baseband freq
    k = 20  # chrip-rate
    n_pts = 100000
    t_min = 0
    t_max = 4 * np.pi
    snr = 1

    noise = np.random.normal(0, snr, n_pts)  # mean, sigma n_pts
    # noise = 0.5*np.random.randn(n_pts)

    t = np.linspace(t_min, t_max, n_pts)

    signal_rx0 = lfm_signal_e(t, [2, 3], f0, k, True)  # signals from different reflections
    signal_rx1 = lfm_signal_e(t, [4, 5], f0, k, True)  # time reversed

    a0 = 0.8  # attenuation factors
    a1 = 0.4

    signal_tx = lfm_signal_e(t, [0, 1], f0, k)
    signal_tx_flipped = lfm_signal_e(t, [0, 1], f0, k, True)
    signal_rx = a0 * signal_rx0# + a1 * signal_rx1
    # signal_rx = a0 * signal_rx1
    signal_rx = a0 * signal_rx0 + a1 * signal_rx1

    signal_rx_noise = signal_rx + noise

    signal_correlated = scipy.signal.correlate(signal_rx, signal_tx_flipped, mode="same")
    signal_correlated_noise = scipy.signal.correlate(signal_rx_noise, signal_tx_flipped, mode="same")

    _, ax = plt.subplots(5, sharex=True, figsize=(10, 20))

    ax[0].plot(t, get_real_iq(signal_tx))
    ax[0].set_title("Transmitted Signal")

    ax[1].plot(t, get_real_iq(signal_rx))
    ax[1].set_title("Reflected Signals")

    ax[2].plot(t, get_real_iq(signal_rx_noise))
    ax[2].set_title("Received Noisy Signal")

    ax[3].plot(t, get_real_iq(signal_correlated))
    ax[3].set_title("Correlated Signal No Noise")

    ax[4].plot(t, get_real_iq(signal_correlated_noise))
    ax[4].set_title("Correlated Signal With Noise")

    for a in ax:
        a.set_xticks([], []) 
        a.set_yticks([], []) 
        a.set_yticklabels([])
        a.set_xticklabels([])

    ax[-1].set_xlabel("Time")
    # plt.plot(t, get_real_iq(signal_rx0))

    plt.show()


if __name__ == "__main__":
    # main()
    # lfm_demo1()
    lfm_demo0()