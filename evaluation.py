import librosa
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from numpy import ndarray

from hparams import hparams


def draw_spectrogram(data: np.ndarray, show=False, dpi=150, **kwargs):
    """

    :param data: db-scale magnitude spectrogram
    :param to_db:
    :param show:
    :param dpi:
    :param kwargs: vmin, vmax
    :return:
    """

    data = data.squeeze()

    fig = plt.figure(dpi=dpi, )
    plt.imshow(data,
               cmap=plt.get_cmap('CMRmap'),
               extent=(0, data.shape[1], 0, hparams.sample_rate // 2),
               origin='lower', aspect='auto', **kwargs)
    plt.xlabel('Frame Index')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    if show:
        plt.show()

    return fig


def draw_roc_curve(y: ndarray, pred_prob: ndarray, show=False):
    fig, ax = plt.subplots(dpi=200)

    skplt.metrics.plot_roc(y, pred_prob, ax=ax)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    if show:
        fig.show()

    return fig


def draw_confusion_mat(y: ndarray, y_est: ndarray, show=False):
    fig, ax = plt.subplots(dpi=200)

    skplt.metrics.plot_confusion_matrix(y, y_est, ax=ax, normalize=True)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    if show:
        fig.show()

    return fig


def reconstruct_wave(*args: ndarray, n_iter=0, n_sample=-1) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag_spectrogram, phase_spectrogram) or (complex_spectrogram,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param n_sample: number of samples of output wave
    :return:
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.core.istft(mag * np.exp(1j * phase), **hparams.kwargs_istft)

        phase = np.angle(librosa.core.stft(wave, **hparams.kwargs_stft))

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.core.istft(spec, **hparams.kwargs_istft, **kwarg_len)

    return wave
