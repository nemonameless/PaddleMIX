# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import inspect
import os
import math
import json
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdiffusers import AutoencoderKL, DDIMScheduler
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params
from paddlenlp.utils.log import logger
from ppdiffusers.initializer import normal_, reset_initialized_parameter, zeros_

from .dit import DiT


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + paddle.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * paddle.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = paddle.distributions.Normal(paddle.zeros_like(x), paddle.ones_like(x)).log_prob(normalized_x) #####
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = paddle.log(cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = paddle.log((1.0 - cdf_min).clip(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        x < -0.999,
        log_cdf_plus,
        paddle.where(x > 0.999, log_one_minus_cdf_min, paddle.log(cdf_delta.clip(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, paddle.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for paddle.exp().
    logvar1, logvar2 = [
        x if isinstance(x, paddle.Tensor) else paddle.to_tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return paddle.mean(tensor, axis=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = paddle.to_tensor(arr)[timesteps].cast('float32')
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + paddle.zeros(broadcast_shape)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class DiTDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        self.model_mean_type = "epsilon" # PREVIOUS_X START_X EPSILON
        self.model_var_type = "learned_range" # LEARNED FIXED_SMALL FIXED_LARGE LEARNED_RANGE
        self.loss_type = "mse" # MSE RESCALED_MSE KL(is_vb) RESCALED_KL(is_vb)

        # self.use_timesteps = set(use_timesteps)
        # self.timestep_map = []
        # last_alpha_cumprod = 1.0
        # new_betas = []
        # for i, alpha_cumprod in enumerate(self.alphas_cumprod):
        #     if i in self.use_timesteps:
        #         new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
        #         last_alpha_cumprod = alpha_cumprod
        #         self.timestep_map.append(i)

        betas = get_named_beta_schedule('linear', 1000)

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )


        self.transformer = DiT(**read_json(model_args.dit_config_file))
        self.transformer_is_pretrained = False

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type

        # self.noise_scheduler = DDPMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     num_train_timesteps=1000,
        #     prediction_type=self.prediction_type,
        # )
        # self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)

        if model_args.image_logging_steps > 0:
            self.eval_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type=self.prediction_type,
            )
            self.eval_scheduler.set_timesteps(model_args.num_inference_steps)

        self.use_ema = model_args.use_ema
        self.noise_offset = model_args.noise_offset
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)
        self.transformer.train()
        self.vae.eval()

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: paddle.Tensor, noise: paddle.Tensor, timesteps: paddle.Tensor) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.transformer.parameters())
            self.model_ema.copy_to(self.transformer)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.transformer)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = paddle.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, latents=None, label_id=None, **kwargs):
        x_start = latents
        timesteps = paddle.randint(0, self.num_timesteps, (latents.shape[0],))
        
        self.vae.eval()
        noise = paddle.randn(latents.shape)
        x_t = self.q_sample(latents, timesteps, noise=noise)

        model_output = self.transformer(x=x_t, t=timesteps, y=label_id) #.sample

        # Get the target for loss depending on the prediction type
        if self.prediction_type == "epsilon": # default
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        if self.loss_type == "mse":
            B, C = x_t.shape[:2]
            assert model_output.shape == [B, C * 2, *x_t.shape[2:]]
            model_output, model_var_values = paddle.split(model_output, 2, axis=1) ###
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = paddle.concat([model_output, model_var_values], axis=1)
            vb_loss = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=x_t,
                    t=timesteps,
                    clip_denoised=False,
                )["output"]

        assert model_output.shape == target.shape == x_start.shape
        mse_loss = mean_flat((target - model_output) ** 2)
        if self.loss_type == "mse":
            loss = mse_loss + vb_loss
        else:
            loss = mse_loss
        return loss

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = paddle.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == [B]
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in ["learned_range"]:
            assert model_output.shape == [B, C * 2, *x.shape[2:]]
            model_output, model_var_values = paddle.split(model_output, 2, axis=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = paddle.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clip(-1, 1) ###
            return x

        if self.model_mean_type == 'start_x':
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, **kwargs):
        self.eval()
        if pixel_values.shape[0] > 8:
            pixel_values = pixel_values[:8]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=256,
        width=256,
        eta=0.0,
        class_labels=[1,2,3,4,5,6,7,8],
        guidance_scale=4.0,
        **kwargs,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 8 image
            if input_ids.shape[0] > 8:
                input_ids = input_ids[:8]

            batch_size = input_ids.shape[0]
            latent_channels = self.transformer.in_channels

            latents = paddle.randn((input_ids.shape[0], self.transformer.in_channels, height // 8, width // 8))
            latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

            class_labels = paddle.to_tensor(class_labels).flatten()
            class_null = paddle.to_tensor([1000] * batch_size)
            class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                if guidance_scale > 1:
                    half = latent_model_input[: len(latent_model_input) // 2]
                    latent_model_input = paddle.concat([half, half], axis=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timesteps = t
                if not paddle.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    if isinstance(timesteps, float):
                        dtype = paddle.float32
                    else:
                        dtype = paddle.int64
                    timesteps = paddle.to_tensor([timesteps], dtype=dtype)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None]
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(
                    [
                        latent_model_input.shape[0],
                    ]
                )
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input, timestep=timesteps, class_labels=class_labels_input
                ).sample

                # perform guidance
                if guidance_scale > 1:
                    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    bs = eps.shape[0]
                    # TODO torch.split vs paddle.split
                    cond_eps, uncond_eps = paddle.split(eps, [bs // 2, bs - bs // 2], axis=0)

                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = paddle.concat([half_eps, half_eps], axis=0)

                    noise_pred = paddle.concat([eps, rest], axis=1)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    # TODO torch.split vs paddle.split
                    model_output, _ = paddle.split(
                        noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                    )
                else:
                    model_output = noise_pred

                # compute previous image: x_t -> x_t-1
                latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

            if guidance_scale > 1:
                latents, _ = latent_model_input.chunk(2, axis=0)
            else:
                latents = latent_model_input

            latents = 1 / self.vae.config.scaling_factor * latents
            samples = self.vae.decode(latents).sample
            samples = (samples / 2 + 0.5).clip(0, 1)
            image = samples.transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, value=False):
        def fn(layer):
            # ldmbert
            if hasattr(layer, "enable_recompute"):
                layer.enable_recompute = value
                print("Set", layer.__class__, "recompute", layer.enable_recompute)
            # unet
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = value
                print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

        self.apply(fn)
