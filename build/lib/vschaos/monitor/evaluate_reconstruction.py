from .evaluate_evalute import Evaluation
from ..utils.misc import decudify
from ..criterions.criterion_spectral import SpectralLoss
from ..criterions.criterion_logdensities import LogDensity
from ..criterions.criterion_functional import MSE, L1


class ReconstructionEvaluation(Evaluation):

    def evaluate_reconstruction(self, vae_out, target, input_params, classwise=False, **kwargs):
        # Evaluate classical reconstruction losses
        reconstruction_loss = LogDensity() + MSE() + L1()
        reduction = kwargs.get('reduction', 'none'); reconstruction_loss.reduction = reduction
        rec_out, rec_losses = reconstruction_loss(params1=vae_out['x_params'], params2=target, input_params=input_params)
        return {**reconstruction_loss.get_named_losses(rec_losses)}

    def evaluate(self, outputs, target=None, model=None, **kwargs):
        reconstruction_losses = self.evaluate_reconstruction(outputs, target, model.pinput, **kwargs)
        return reconstruction_losses


class SpectralEvaluation(ReconstructionEvaluation):

    def parse_params(self, spectral_params=None, **kwargs):
        super(SpectralEvaluation, self).parse_params(**kwargs)
        assert spectral_params
        self.spectral_params = spectral_params

    def evaluate_reconstruction(self, vae_out, target, input_params, **kwargs):
        # Evaluate classical reconstruction losses
        rec_losses = super(SpectralEvaluation, self).evaluate_reconstruction(vae_out, target, input_params, **kwargs)
        # Evaluate spectral losses
        spectral_loss = SpectralLoss(preprocessing=kwargs.get('preprocessing'), **self.spectral_params)
        reduction = kwargs.get('reduction', 'none'); spectral_loss.reduction = reduction
        spectral_out, spectral_losses = spectral_loss(vae_out['x_params'], target, input_params=input_params)
        return {**rec_losses, **spectral_loss.get_named_losses(spectral_losses)}


class PredictionEvaluation(ReconstructionEvaluation):

    def parse_params(self, evaluations=[], **kwargs):
        self.evaluations = evaluations

    def evaluate_reconstruction(self, vae_out, target, input_params, predict=False, **kwargs):
        prediction = vae_out.get('prediction')
        if prediction is None or predict is False:
            raise UserWarning('prediction not found! (predict is: %s)'%predict)
        prediction_length = kwargs.get('n_preds', prediction['out'].shape[1])
        x_out = vae_out['x_params'][:, -prediction_length:]
        target = target[:, -prediction_length:]
        reconstruction_loss = LogDensity() + MSE() + L1()
        reduction = kwargs.get('reduction', 'none'); reconstruction_loss.reduction = reduction
        rec_out, rec_losses = reconstruction_loss(x_params=x_out, target=target, input_params=input_params)
        return {'prediction': {**reconstruction_loss.get_named_losses(rec_losses)}}




class RawEvaluation(Evaluation):

    def parse_params(self, eval_params):
        self.reconstruction_params = eval_params.get('reconstruction')
        self.spectral_params = eval_params.get('spectral')
        self.latent_params = eval_params.get('latent')

    def forward_model(self, model, loader, **kwargs):
        outs = []; target = []
        preprocessing = kwargs.get('preprocessing')
        for x, y in loader:
            if preprocessing:
                x = preprocessing(x)
            x = model.format_input_data(x)
            outs.append(decudify(model.forward(x, y=y, **kwargs)))
            target.append(x)
        outs_merged = merge_dicts(outs)
        target_merged = merge_dicts(target)
        return outs_merged, target_merged

    def evaluate(self, outputs, target=None, model=None, **kwargs):
        # spectral loss
        spectral_loss = SpectralLoss(**self.spectral_params)
        spectral_losses = spectral_loss.get_named_losses(spectral_loss(outputs['x_params'], target, plot=False)[1])
        # reconstruction loss
        reconstruction_loss = LogDensity()+MSE()
        if model is not None:
            input_params = model.pinput
        reconstruction_losses = reconstruction_loss.get_named_losses(reconstruction_loss(x_params=outputs['x_params'], target=target, input_params=input_params)[1])

        return {**spectral_losses, **reconstruction_losses}





