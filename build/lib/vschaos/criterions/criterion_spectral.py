import numpy as np, pdb, math
import matplotlib.pyplot as plt
import torch, librosa.filters, librosa.feature
import torch.nn as nn
from . import Criterion
from .. import distributions as dist

eps = 1e-5
log_norm_order = 2

def wrap(angles):
    return angles

# Raw spectral loss functions
def spectral_convergence(*, x_radius=None, target_radius=None, **kwargs):
    x = x_radius; target = target_radius
    frobenius_diff = torch.sqrt(torch.sum(torch.pow(target - x, 2), 1))
    frobenius_input_amp = torch.clamp(torch.sqrt(torch.sum(torch.pow(target, 2), 1)), eps, None)
    spectral_convergence = torch.mean(frobenius_diff / frobenius_input_amp)
    return spectral_convergence

def log_magnitude_difference(*, x_radius=None, target_radius=None, **kwargs):
    x = x_radius; target = target_radius
    log_stft_mag_diff = torch.sum(torch.abs(torch.log(target + eps) - torch.log(x + eps)), 1)
    log_stft_ref = torch.sum(torch.abs(torch.log(target + eps)), 1)
    log_stft_mag = torch.mean(log_stft_mag_diff/(log_stft_ref + eps))
    return log_stft_mag

def l2_mag(*, x_radius=None, target_radius=None, **kwargs):
    return nn.functional.mse_loss(x_radius, target_radius)


def log_isd(*, x_radius=None, target_radius=None, **kwargs):
    x = x_radius+eps; target = target_radius+eps
    return torch.log((1 / (2 * np.pi)) * ((target / x).sum() - torch.log(target/x).sum() - 1))/x_radius.shape[0]

def weighted_phase(*, x_real=None, target_real=None, x_radius=None, target_radius=None, x_imag=None, target_imag=None, **kwargs):
    return torch.sum(target_radius * x_radius - (target_real * x_real) - (target_imag * x_imag))

def log_norm_mag(*, x_radius=None, target_radius=None, **kwargs):
    return nn.functional.mse_loss(x_radius.log(), target_radius.clamp(1e-5).log())
    #return torch.sum(torch.pow(torch.abs(torch.pow(torch.log(x_radius), log_norm_order) - torch.pow(torch.log(target_radius), log_norm_order)), 1/log_norm_order))

def log_norm_angle(*, x_angle=None, target_angle=None, **kwargs):
    x = wrap(x_angle)
    target = wrap(target_angle)
    return torch.sum(torch.pow(torch.abs(torch.pow(x, log_norm_order) - torch.pow(target, log_norm_order)), 1/log_norm_order))


class InstantaneousFrequency(nn.Module):
    """ Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_. This class is a PyTorch transcription of the librosa.ifgram
    function.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Parameters
    ----------

    size: int
        Size of the fft, window, and hop_length = size // 4
    """

    def __init__(self, size):
        super(InstantaneousFrequency, self).__init__()

        self.size    = size
        self.hop     = size //4
        self.window  = nn.Parameter(torch.hann_window(size))

        freq_angular = np.linspace(0, 2 * np.pi, size, endpoint=False)
        d_window = np.sin(-freq_angular) * np.pi / size
        self.d_window = nn.Parameter(torch.from_numpy(d_window).float())

    def magphase(self, x):
        """ Returns the magnitude and phase of a stft """
        mag   = torch.norm(x, dim=3)
        phase = torch.atan2(x[:,:,:,1],x[:,:,:,0]+1e-15)
        return mag,phase

    def forward(self, x):
        """ Forward pass of the module

        Parameters
        ----------

        x: Tensor
            Input signal of which IF is to be computed. Must be of size [BxW]
        """
        stft_matrix = torch.stft(x, n_fft=self.size, hop_length=self.hop,
                               win_length=self.size,
                               window=self.window)
        diff_stft   = torch.stft(x, n_fft=self.size, hop_length=self.hop,
                               window=self.d_window)
        diff_stft[:,:,:,1] *= -1
        mag, phase = self.magphase(stft_matrix)
        bin_offset = (- phase * diff_stft[:,:,:,1]) / (mag + 1e-3)
        return bin_offset


def log_norm_inst(*, x=None, target=None, x_spectral=None, target_spectral=None):
    converter = InstantaneousFrequency(256)
    converter = converter.to(device=x.device)
    x = converter(x.squeeze())
    target = converter(target.squeeze())
    return 10 * torch.sum(torch.pow(torch.abs(torch.pow(x, log_norm_order) - torch.pow(target, log_norm_order)), 1/log_norm_order))


#TODO probabilistic log-likelihoods with Normal distriution for amplitude / Von-Mises distriuition?
class SpectralLoss(Criterion):
    losses_hash = {'spec_conv':spectral_convergence, 'log_diff':log_magnitude_difference, 'log_isd':log_isd, 'l2_mag':l2_mag,
            'weighted_phase':weighted_phase, 'log_mag':log_norm_mag, 'log_phase':log_norm_angle}
    energy_threshold = 1e-4

    def __init__(self, n_fft=256, hop_length=None, win_length=None, transform = "stft", losses=None, weights=None, is_sequence=False,
                 mel_filters=None, preprocessing=None, high_cut=0, spectral_warmup=0, plot_period=50, figures_folder='', **kwargs):
        super(SpectralLoss, self).__init__(**kwargs)
        assert torch.backends.mkl.is_available(), "spectral loss need MKL module to be used...!"
        # spectral transform used
        self.transform = transform
        self.preprocessing = preprocessing
        # reconstruction callbacks
        self.spec_losses = losses if losses else ['log_diff']
        # balance between magnitude / phase losses
        self.weights = weights or [1.0]*len(self.spec_losses)
        # spectral parameters
        self.spectral_args = {'n_fft':n_fft, 'hop_length':hop_length, 'win_length':win_length}
        self.mel_filters = mel_filters or 512
        self.high_cut = high_cut
        self.spectral_warmup = spectral_warmup
        self.figures_folder = figures_folder
        self.plot_period = plot_period
        self.take_sequences = is_sequence

    def get_transform(self, x, transform=None):
        transform = transform or self.transform
        window_size = self.spectral_args.get('win_length') or self.spectral_args.get('n_fft')
        if transform in ('stft', 'stft-mel'):
            x_fft = torch.stft(x.squeeze(), window=torch.hann_window(window_size, device=x.device), center=True, pad_mode='constant', **self.spectral_args)
            x_fft_real = x_fft[:,:,:, 0]; x_fft_imag = x_fft[:,:,:, 1];
        elif transform in ('fft', 'fft-mel'):
            x_fft = torch.fft(torch.cat([x.squeeze().unsqueeze(-1), torch.zeros_like(x.squeeze().unsqueeze(-1))], dim=-1), 2)
            x_fft_real = x_fft.select(-1, 0); x_fft_imag = x_fft.select(-1, 1);
            x_fft_real = x_fft_real[:, :int(x_fft_real.shape[1]/2+1)];
            x_fft_imag = x_fft_imag[:, :int(x_fft_imag.shape[1]/2+1)];
            window_size = x_fft_real.shape[1]*2

        x_radius = torch.sqrt(x_fft_real.pow(2) + x_fft_imag.pow(2))
        x_angle = torch.atan2(x_fft_real, x_fft_imag+eps)
        if transform in ("stft-mel", 'fft-mel'):
            mel_w = librosa.filters.mel(22050, window_size-1, n_mels = min(self.mel_filters, window_size))
            mel_weights = torch.from_numpy(mel_w).float().to(x_fft).detach()
            x_radius = torch.bmm(mel_weights.unsqueeze(0).repeat(x_radius.shape[0],1,1), x_radius.unsqueeze(-1)).transpose(1,2)
            if transform == "fft-mel":
                x_radius = x_radius.squeeze()

        return x_fft_real.unsqueeze(1), x_fft_imag.unsqueeze(1), x_radius.unsqueeze(1), x_angle.unsqueeze(1)


    def get_valid_ids(self, x):
        x = x.squeeze().cpu().detach().numpy()
        print(type(x), x.shape)
        energies = [librosa.feature.rmse(y=x[i], frame_length=min(x.shape[-1], 1024), center=False) for i in range(x.shape[0])]
        ids = np.where(np.array(energies) >= self.energy_threshold)[0]
        return ids

    def loss(self, x_params, target, input_params=None, preprocessing=None, sample=False, losses=None,
             transform=None, weights=None, plot=False, period=None, *args, **kwargs):
        if torch.cuda.is_available():
            torch.backends.cuda.cufft_plan_cache.clear()
        spec_losses = losses or self.spec_losses
        transform = transform or self.transform
        if not weights is None:
            assert len(weights) == len(spec_losses)
        weights = weights or self.weights or [1.0]*len(spec_losses)

        if sample:
            x_valued = x_params.rsample() if x_params.has_rsample else x_params.sample()
        else:
            if issubclass(type(x_params), (dist.Normal, dist.priors.WienerProcess, dist.Bernoulli)):
                x_valued = x_params.mean
            elif issubclass(type(x_params), dist.Categorical):
                x_probs = x_params.probs
                x_valued = (x_probs >= torch.max(x_probs, dim=1)[0].unsqueeze(1).repeat(1, x_probs.shape[1], 1)).float()

        # preprocessing in case
        preprocessing = preprocessing or self.preprocessing or None
        if preprocessing:
            target = preprocessing.invert(target)
            x_valued = preprocessing.invert(x_valued)


        batch_shape = target.shape[0]
        # computing transforms
        if self.take_sequences:
            x_valued = x_valued.contiguous().view(x_valued.shape[0]*x_valued.shape[1], *x_valued.shape[2:])
            target = target.contiguous().view(target.shape[0]*target.shape[1], *target.shape[2:])

        if transform is not None:
            x_real, x_imag, x_radius, x_angle = self.get_transform(x_valued, transform=transform)
            target_real, target_imag, target_radius, target_angle = self.get_transform(target.float(), transform=transform)
        else:
            x_radius = x_valued; target_radius = target.float()
            x_real = x_imag = x_angle = None
            target_real = target_imag = target_angle = None


        # computing losses
        losses = []; loss = 0
        for i, l in enumerate(self.spec_losses):
            current_loss = self.losses_hash[l](x_radius=x_radius, target_radius = target_radius,
                                            x_angle =x_angle, target_angle = target_angle,
                                            x_real = x_real, target_real = target_real,
                                            x_imag = x_imag, target_imag = target_imag)
            loss = loss + weights[i] * current_loss
            losses.append(current_loss.detach().cpu().numpy())
        loss = loss / batch_shape


        # spectral warm-up
        if self.spectral_warmup > 0 and kwargs.get('epoch') is not None:
            loss = loss * min(kwargs['epoch'] / self.spectral_warmup, 1.0)

        # plotting in case
        epoch = kwargs.get('epoch', 0)
        if plot and epoch % self.plot_period == 0:
            n_random_examples = min(5, x_valued.shape[0])
            ids = np.random.permutation(x_valued.shape[0])[:n_random_examples]
            target_pl = target_radius[ids].squeeze(); x_pl = x_radius[ids].squeeze()
            plot_name = "spectrum_%s"%period
            if len(target_pl.shape) > 2:
                fig, ax = plt.subplots(n_random_examples, 2)
                for i in range(n_random_examples):
                    ax[i,0].imshow(target_pl[i].log().cpu().detach().numpy(), aspect="auto")
                    ax[i,1].imshow(x_pl[i].log().cpu().detach().numpy(), aspect="auto")
                fig.savefig(self.figures_folder+'/%s_log_%s.pdf'%(plot_name, kwargs.get('epoch')), format='pdf')
            else:
                fig, ax = plt.subplots(n_random_examples, 1)
                for i in range(n_random_examples):
                    ax[i].plot(target_pl[i].log().cpu().detach().numpy())
                    ax[i].plot(x_pl[i].log().cpu().detach().numpy(), linewidth=0.7)
                fig.savefig(self.figures_folder+'/%s_log_%s.pdf'%(plot_name, kwargs.get('epoch')), format='pdf')

            if len(target_pl.shape) > 2:
                fig, ax = plt.subplots(n_random_examples, 2)
                for i in range(n_random_examples):
                    ax[i,0].imshow(target_pl[i].cpu().detach().numpy(), aspect="auto")
                    ax[i,1].imshow(x_pl[i].cpu().detach().numpy(), aspect="auto")
                fig.savefig(self.figures_folder+'/%s_%s.pdf'%(plot_name, kwargs.get('epoch')), format='pdf')
            else:
                fig, ax = plt.subplots(n_random_examples, 1)
                for i in range(n_random_examples):
                    ax[i].plot(target_pl[i].cpu().detach().numpy())
                    ax[i].plot(x_pl[i].cpu().detach().numpy(), linewidth=0.7)
                fig.savefig(self.figures_folder+'/%s_%s.pdf'%(plot_name, kwargs.get('epoch')), format='pdf')

        return loss, tuple(losses)

    def get_named_losses(self, losses):
        return {self.spec_losses[l]: losses[l] for l in range(len(self.spec_losses))}







