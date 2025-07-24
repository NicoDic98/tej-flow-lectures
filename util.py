import matplotlib.pyplot as plt
import numpy as np

### PLOTTING
pix_to_pt = 72/100

basic_config = {
    'lines.linewidth': 1.0 * pix_to_pt,
    'lines.markeredgewidth': 1.0 * pix_to_pt,
    'savefig.bbox': 'tight',
    'axes.labelsize': 9,
    'axes.linewidth': 0.5 * pix_to_pt,
    'lines.markersize': 2,
    'errorbar.capsize': 1,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.5 * pix_to_pt,
    'xtick.minor.width': 0.5 * pix_to_pt,
    'ytick.major.width': 0.5 * pix_to_pt,
    'ytick.minor.width': 0.5 * pix_to_pt,
    'legend.fontsize': 8,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': '0.8',
    'legend.facecolor': 'white',
    'legend.edgecolor': '0.5',
    'patch.linewidth': 0.5 * pix_to_pt,
    'legend.handlelength': 1.0,
    'legend.fontsize': 8,
    'font.family': 'serif'
}

def load_basic_config():
    """Set default matplotlib settings to make plots not horrible."""
    plt.rcParams.update(basic_config)


### ANALYSIS
mean = lambda x: np.mean(x, axis=0)
rmean = lambda x: np.real(np.mean(x, axis=0))
imean = lambda x: np.imag(np.mean(x, axis=0))

def bootstrap_gen(*samples, Nboot, seed=None):
    rng = np.random.default_rng(seed=seed)
    n = len(samples[0])
    for i in range(Nboot):
        inds = rng.integers(n, size=n)
        yield tuple(s[inds] for s in samples)

def bootstrap(*samples, Nboot: int, f, bias_correction=False, seed=None):
    """
    Estimate bootstrap mean and uncertainty for the value of `f(*samples)`.

     - f: The function to apply. Usually something like `np.mean(x, axis=0)`.
     - Nboot: Number of bootstrap samples to draw
     - seed: If not `None`, fixed seed for bootstrap samples
    """
    boots = []
    for x in bootstrap_gen(*samples, Nboot=Nboot, seed=seed):
        boots.append(f(*x))
    boot_mean, boot_err = np.mean(boots, axis=0), np.std(boots, axis=0)
    if not bias_correction:
        return boot_mean, boot_err
    full_mean = f(*samples)
    corrected_mean = 2*full_mean - boot_mean
    return corrected_mean, boot_err

def bin_data(x, *, binsize, silent_trunc=True):
    x = np.array(x)
    if silent_trunc:
        x = x[:(x.shape[0] - x.shape[0]%binsize)]
    else:
        assert x.shape[0] % binsize == 0
    ts = np.arange(0, x.shape[0], binsize) # left endpoints of bins
    x = np.reshape(x, (-1, binsize) + x.shape[1:])
    return ts, np.mean(x, axis=1)
