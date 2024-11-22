#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


# Standard Library
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third Party
import numpy as np
import scipy.interpolate as si
import torch
import torch.autograd.profiler as profiler
from scipy.stats.qmc import Halton
from torch.distributions.multivariate_normal import MultivariateNormal

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.util.logger import log_error, log_warn
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from ..opt.particle.particle_opt_utils import get_stomp_cov


@dataclass
class SampleConfig:
    horizon: int
    d_action: int
    tensor_args: TensorDeviceType
    fixed_samples: bool = True
    sample_ratio: Dict[str, float] = field(
        default_factory=lambda: (
            {"halton": 0.3, "halton-knot": 0.7, "random": 0.0, "random-knot": 0.0, "stomp": 0.0}
        )
    )
    seed: int = 0
    filter_coeffs: Optional[List[float]] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    n_knots: int = 3
    scale_tril: Optional[float] = None
    covariance_matrix: Optional[torch.tensor] = None
    sample_method: str = "halton"
    cov_mode: str = "vel"  # for STOMP sampler
    sine_period: int = 2  # for Sinewave sampler
    degree: int = 3  # bspline


class BaseSampleLib(SampleConfig):
    @profiler.record_function("sample_lib/init")
    def __init__(
        self,
        sample_config,
    ):
        super().__init__(**vars(sample_config))

        self.Z = torch.zeros(
            self.horizon * self.d_action,
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )
        if self.scale_tril is None and self.covariance_matrix is not None:
            self.scale_tril = self.tensor_args.to_device(
                torch.linalg.cholesky(covariance_matrix.to("cpu"))
            )
        self.samples = None
        self.sample_shape = 0
        self.ndims = self.horizon * self.d_action
        self.stomp_matrix, self.stomp_scale_tril = None, None

    def get_samples(self, sample_shape, base_seed, current_state=None, **kwargs):
        raise NotImplementedError

    def filter_samples(self, eps):
        if self.filter_coeffs is not None:
            beta_0, beta_1, beta_2 = self.filter_coeffs

            # This could be tensorized:
            for i in range(2, eps.shape[1]):
                eps[:, i, :] = (
                    beta_0 * eps[:, i, :] + beta_1 * eps[:, i - 1, :] + beta_2 * eps[:, i - 2, :]
                )
        return eps

    def filter_smooth(self, samples):
        # scale by stomp matrix:
        if samples.shape[0] == 0:
            return samples
        if self.stomp_matrix is None:
            self.stomp_matrix, self.stomp_scale_tril = get_stomp_cov(
                self.horizon, self.d_action, tensor_args=self.tensor_args
            )

        # fit bspline:

        filter_samples = self.stomp_matrix[: self.horizon, : self.horizon] @ samples
        # print(filter_samples.shape)
        filter_samples = filter_samples / torch.max(torch.abs(filter_samples))
        return filter_samples


class HaltonSampleLib(BaseSampleLib):
    @profiler.record_function("sample_lib/halton")
    def __init__(self, sample_config: SampleConfig):
        super().__init__(sample_config)
        # create halton generator:
        self.halton_generator = HaltonGenerator(
            self.d_action, seed=self.seed, tensor_args=self.tensor_args
        )

    def get_samples(self, sample_shape, base_seed=None, filter_smooth=False, **kwargs):
        if self.sample_shape != sample_shape or not self.fixed_samples:
            if len(sample_shape) > 1:
                log_error("sample shape should be a single value")
                raise ValueError
            seed = self.seed if base_seed is None else base_seed
            self.sample_shape = sample_shape
            self.seed = seed
            self.samples = self.halton_generator.get_gaussian_samples(
                sample_shape[0] * self.horizon
            )
            self.samples = self.samples.view(sample_shape[0], self.horizon, self.d_action)

            if filter_smooth:
                self.samples = self.filter_smooth(self.samples)
            else:
                self.samples = self.filter_samples(self.samples)
        if self.samples.shape[0] != sample_shape[0]:
            log_error("sampling failed")
        return self.samples


def bspline(c_arr: torch.Tensor, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()

    if t_arr is None:
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)

    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)

    return samples


class KnotSampleLib(SampleConfig):
    def __init__(self, sample_config: SampleConfig):
        super().__init__(**vars(sample_config))
        self.sample_shape = 0
        self.ndims = self.n_knots * self.d_action
        self.Z = torch.zeros(
            self.ndims, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        if self.covariance_matrix is None:
            self.cov_matrix = torch.eye(
                self.ndims, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        self.scale_tril = torch.linalg.cholesky(
            self.cov_matrix.to(dtype=torch.float32, device="cpu")
        ).to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        if self.sample_method == "random":
            self.mvn = MultivariateNormal(
                loc=self.Z,
                scale_tril=self.scale_tril,
            )
        if self.sample_method == "halton":
            self.halton_generator = HaltonGenerator(
                self.ndims, seed=self.seed, tensor_args=self.tensor_args
            )

    def get_samples(self, sample_shape, **kwargs):
        if self.sample_shape != sample_shape or not self.fixed_samples:
            # sample shape is the number of particles to sample
            if self.sample_method == "halton":
                self.knot_points = self.halton_generator.get_gaussian_samples(sample_shape[0])
            elif self.sample_method == "random":
                self.knot_points = self.mvn.sample(sample_shape=sample_shape)

            # Sample splines from knot points:
            # iteratre over action dimension:
            knot_samples = self.knot_points.view(sample_shape[0], self.d_action, self.n_knots)
            self.samples = torch.zeros(
                (sample_shape[0], self.horizon, self.d_action),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            for i in range(sample_shape[0]):
                for j in range(self.d_action):
                    self.samples[i, :, j] = bspline(
                        knot_samples[i, j, :], n=self.horizon, degree=self.degree
                    )
            self.sample_shape = sample_shape

        return self.samples


class RandomSampleLib(BaseSampleLib):
    def __init__(self, sample_config: SampleConfig):
        super().__init__(sample_config)

        if self.scale_tril is None:
            self.scale_tril = torch.eye(
                self.ndims, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

        self.mvn = MultivariateNormal(loc=self.Z, scale_tril=self.scale_tril)

    def get_samples(self, sample_shape, base_seed=None, filter_smooth=False, **kwargs):
        if base_seed is not None and base_seed != self.seed:
            self.seed = base_seed
            # print(self.seed)
            torch.manual_seed(self.seed)
        if self.sample_shape != sample_shape or not self.fixed_samples:
            self.sample_shape = sample_shape
            self.samples = self.mvn.sample(sample_shape=self.sample_shape)
            self.samples = self.samples.view(self.samples.shape[0], self.horizon, self.d_action)
            if filter_smooth:
                self.samples = self.filter_smooth(self.samples)
            else:
                self.samples = self.filter_samples(self.samples)
        return self.samples


class SineSampleLib(BaseSampleLib):  # pragma : no cover
    def __init__(self, sample_config: SampleConfig):
        super().__init__(sample_config)

        self.const_pi = torch.acos(torch.zeros(1)).item()
        self.ndims = self.d_action
        self.sine_wave = self.generate_sine_wave()
        self.diag_sine_wave = torch.diag(self.sine_wave)

    def get_samples(self, sample_shape, base_seed=None, **kwargs):  # pragma : no cover
        if self.sample_shape != sample_shape or not self.fixed_samples:
            if len(sample_shape) > 1:
                print("sample shape should be a single value")
                raise ValueError
            seed = self.seed if base_seed is None else base_seed
            self.sample_shape = sample_shape
            self.seed = seed

            # sample only amplitudes from halton sequence:
            self.amplitude_samples = generate_gaussian_halton_samples(
                sample_shape[0],
                self.ndims,
                use_scipy_halton=True,
                seed=self.seed,
                tensor_args=self.tensor_args,
            )

            self.amplitude_samples = self.filter_samples(self.amplitude_samples)
            self.amplitude_samples = self.amplitude_samples.unsqueeze(1).expand(
                -1, self.horizon, -1
            )

            # generate sine waves from samples for the full horizon:
            self.samples = self.diag_sine_wave @ self.amplitude_samples

        return self.samples

    def generate_sine_wave(self, horizon=None):  # pragma : no cover
        horizon = self.horizon if horizon is None else horizon

        # generate a sine wave:
        x = torch.linspace(
            0,
            4 * self.const_pi / self.sine_period,
            horizon,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        sin_out = torch.sin(x)

        return sin_out


class StompSampleLib(BaseSampleLib):
    @profiler.record_function("stomp_sample_lib/init")
    def __init__(
        self,
        sample_config: SampleConfig,
    ):
        super(StompSampleLib, self).__init__(sample_config)

        _, self.stomp_scale_tril, _ = get_stomp_cov(
            self.horizon,
            self.d_action,
            tensor_args=self.tensor_args,
            cov_mode=self.cov_mode,
            RETURN_M=True,
        )

        self.filter_coeffs = None
        self.halton_generator = HaltonGenerator(
            self.d_action, seed=self.seed, tensor_args=self.tensor_args
        )

    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        if self.sample_shape != sample_shape or not self.fixed_samples:
            if len(sample_shape) > 1:
                print("sample shape should be a single value")
                raise ValueError
            # seed = self.seed if base_seed is None else base_seed
            self.sample_shape = sample_shape

            # self.seed = seed
            # torch.manual_seed(self.seed)
            halton_samples = self.halton_generator.get_gaussian_samples(
                sample_shape[0] * self.horizon
            ).view(sample_shape[0], self.horizon * self.d_action)

            halton_samples = (
                self.stomp_scale_tril.unsqueeze(0) @ halton_samples.unsqueeze(-1)
            ).squeeze(-1)

            halton_samples = (
                (halton_samples)
                .view(self.sample_shape[0], self.d_action, self.horizon)
                .transpose(-2, -1)
            )
            halton_samples = halton_samples / torch.max(torch.abs(halton_samples))
            # halton_samples[:, 0, :] = 0.0
            halton_samples[:, -1:, :] = 0.0
            if torch.any(torch.isnan(halton_samples)):
                log_error("Nan values found in samplelib, installation could have been corrupted")
            self.samples = halton_samples
        return self.samples


class SampleLib(BaseSampleLib):
    def __init__(self, sample_config: SampleConfig):
        super().__init__(sample_config)
        # sample from a mix of possibilities:
        # TODO: Create instances only if the ratio is not 0.0
        # halton
        self.halton_sample_lib = HaltonSampleLib(sample_config)
        self.knot_halton_sample_lib = KnotSampleLib(sample_config)
        self.random_sample_lib = RandomSampleLib(sample_config)
        self.knot_random_sample_lib = KnotSampleLib(sample_config)
        self.stomp_sample_lib = StompSampleLib(sample_config)
        self.sine_sample_lib = SineSampleLib(sample_config)
        self.sample_fns = []

        self.sample_fns = {
            "halton": self.halton_sample_lib.get_samples,
            "halton-knot": self.knot_halton_sample_lib.get_samples,
            "random": self.random_sample_lib.get_samples,
            "random-knot": self.knot_random_sample_lib.get_samples,
            "stomp": self.stomp_sample_lib.get_samples,
            "sine": self.sine_sample_lib.get_samples,
        }
        self.samples = None

    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        # TODO: Make sure ratio * sample_shape is an integer

        if (
            (not self.fixed_samples)
            or self.samples is None
            or sample_shape[0] != self.samples.shape[0]
        ):
            cat_list = []
            sample_shape = list(sample_shape)
            for ki, k in enumerate(self.sample_ratio.keys()):
                if self.sample_ratio[k] == 0.0:
                    continue
                n_samples = round(sample_shape[0] * self.sample_ratio[k])
                s_shape = torch.Size([n_samples])
                # if(k == 'halton' or k == 'random'):
                samples = self.sample_fns[k](sample_shape=s_shape)
                # else:
                #    samples = self.sample_fns[k](sample_shape=s_shape)
                cat_list.append(samples)
                samples = torch.cat(cat_list, dim=0)
                self.samples = samples
        return self.samples


def get_ranged_halton_samples(
    dof,
    up_bounds,
    low_bounds,
    num_particles,
    tensor_args: TensorDeviceType = TensorDeviceType("cpu"),
    seed=123,
):
    q_samples = generate_halton_samples(
        num_particles,
        dof,
        use_scipy_halton=True,
        tensor_args=tensor_args,
        seed=seed,
    )

    # scale samples by joint range:
    range_b = up_bounds - low_bounds
    q_samples = q_samples * range_b + low_bounds

    return q_samples


class HaltonGenerator:
    def __init__(
        self,
        ndims,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        up_bounds=[1],
        low_bounds=[0],
        seed=123,
        store_buffer: Optional[int] = 2000,
    ):
        self._seed = seed
        self.tensor_args = tensor_args
        self.sequencer = Halton(d=ndims, seed=seed, scramble=False)
        # scale samples by joint range:
        up_bounds = self.tensor_args.to_device(up_bounds)
        low_bounds = self.tensor_args.to_device(low_bounds)
        self.range_b = up_bounds - low_bounds
        self.low_bounds = low_bounds
        self.ndims = ndims
        self.proj_mat = torch.sqrt(
            torch.tensor([2.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        )
        self.i_mat = torch.eye(
            self.ndims, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._sample_buffer = None
        self._store_buffer = store_buffer
        if store_buffer is not None:
            # sample some and just randomly get tensors from this buffer:
            self._sample_buffer = torch.tensor(
                self.sequencer.random(store_buffer),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._int_gen = torch.Generator(device=self.tensor_args.device)
            self._int_gen.manual_seed(seed)
            self._index_buffer = None

    def reset(self):
        self.sequencer.reset()
        if self._store_buffer is not None:
            self._sample_buffer = torch.tensor(
                self.sequencer.random(self._store_buffer),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._int_gen = torch.Generator(device=self.tensor_args.device)
            self._int_gen.manual_seed(self._seed)
            # self._index_buffer = None

    def fast_forward(self, steps: int):
        """
        Fast forward sampler by steps
        """
        self.sequencer.fast_forward(steps)
        if self.fixed_samples:
            log_warn("fast forward will not work with fixed samples.")

    def _get_samples(self, num_samples: int):
        if self._sample_buffer is not None:
            out_buffer = None
            if self._index_buffer is not None and self._index_buffer.shape[0] == num_samples:
                out_buffer = self._index_buffer
            index = torch.randint(
                0,
                self._sample_buffer.shape[0],
                (num_samples,),
                generator=self._int_gen,
                device=self.tensor_args.device,
                out=out_buffer,
            )
            samples = self._sample_buffer[index]
            if self._index_buffer is None:
                self._index_buffer = index
        else:
            samples = torch.tensor(
                self.sequencer.random(num_samples),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        return samples

    @profiler.record_function("halton_generator/samples")
    def get_samples(self, num_samples, bounded=False):
        samples = self._get_samples(num_samples)
        if bounded:
            samples = bound_samples(samples, self.range_b, self.low_bounds)
        return samples

    @profiler.record_function("halton_generator/gaussian_samples")
    def get_gaussian_samples(self, num_samples, variance=1.0):
        std_dev = np.sqrt(variance)
        uniform_samples = self.get_samples(num_samples, False)
        gaussian_halton_samples = gaussian_transform(
            uniform_samples, self.proj_mat, self.i_mat, std_dev
        )
        return gaussian_halton_samples


@get_torch_jit_decorator()
def bound_samples(samples: torch.Tensor, range_b: torch.Tensor, low_bounds: torch.Tensor):
    samples = samples * range_b + low_bounds
    return samples


@get_torch_jit_decorator()
def gaussian_transform(
    uniform_samples: torch.Tensor, proj_mat: torch.Tensor, i_mat: torch.Tensor, std_dev: float
):
    """Compute a guassian transform of uniform samples.

    Args:
        uniform_samples (torch.Tensor): uniform samples in the range [0,1].
        proj_mat (torch.Tensor): _description_
        i_mat (torch.Tensor): _description_
        variance (float): _description_

    Returns:
        _type_: _description_
    """
    # since erfinv returns inf when value is -1 or +1, we scale the input to not have
    # these values.
    changed_samples = 1.99 * uniform_samples - 0.99
    gaussian_halton_samples = proj_mat * torch.erfinv(changed_samples)
    i_mat = i_mat * std_dev
    gaussian_halton_samples = torch.matmul(gaussian_halton_samples, i_mat)
    return gaussian_halton_samples


#######################
## Gaussian Sampling ##
#######################


def generate_noise(cov, shape, base_seed, filter_coeffs=None, device=torch.device("cpu")):
    """
    Generate correlated Gaussian samples using autoregressive process
    """
    torch.manual_seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    m = MultivariateNormal(loc=torch.zeros(N).to(device), covariance_matrix=cov)
    eps = m.sample(sample_shape=shape)
    # eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    if filter_coeffs is not None:
        for i in range(2, eps.shape[1]):
            eps[:, i, :] = (
                beta_0 * eps[:, i, :] + beta_1 * eps[:, i - 1, :] + beta_2 * eps[:, i - 2, :]
            )
    return eps


def generate_noise_np(cov, shape, base_seed, filter_coeffs=None):
    """
    Generate correlated noisy samples using autoregressive process
    """
    np.random.seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov=cov, size=shape)
    if filter_coeffs is not None:
        for i in range(2, eps.shape[1]):
            eps[:, i, :] = (
                beta_0 * eps[:, i, :] + beta_1 * eps[:, i - 1, :] + beta_2 * eps[:, i - 2, :]
            )
    return eps


###########################
## Quasi-Random Sampling ##
###########################


def generate_prime_numbers(num):
    def is_prime(n):
        for j in range(2, ((n // 2) + 1), 1):
            if n % j == 0:
                return False
        return True

    primes = [0] * num  # torch.zeros(num, device=device)
    primes[0] = 2
    curr_num = 1
    for i in range(1, num):
        while True:
            curr_num += 2
            if is_prime(curr_num):
                primes[i] = curr_num
                break

    return primes


def generate_van_der_corput_sample(idx, base):
    f, r = 1.0, 0
    while idx > 0:
        f /= base * 1.0
        r += f * (idx % base)
        idx = idx // base
    return r


def generate_van_der_corput_samples_batch(idx_batch, base):
    inp_device = idx_batch.device
    batch_size = idx_batch.shape[0]
    f = 1.0  # torch.ones(batch_size, device=inp_device)
    r = torch.zeros(batch_size, device=inp_device)
    while torch.any(idx_batch > 0):
        f /= base * 1.0
        r += f * (idx_batch % base)  # * (idx_batch > 0)
        idx_batch = idx_batch // base
    return r


def generate_halton_samples(
    num_samples,
    ndims,
    bases=None,
    use_scipy_halton=True,
    seed=123,
    tensor_args: TensorDeviceType = TensorDeviceType(),
):
    if not use_scipy_halton:
        samples = torch.zeros(
            num_samples, ndims, device=tensor_args.device, dtype=tensor_args.dtype
        )
        if not bases:
            bases = generate_prime_numbers(ndims)
        idx_batch = torch.arange(1, num_samples + 1, device=tensor_args.device)
        for dim in range(ndims):
            samples[:, dim] = generate_van_der_corput_samples_batch(idx_batch, bases[dim])
    else:
        sequencer = Halton(d=ndims, seed=seed, scramble=False)
        samples = torch.tensor(
            sequencer.random(num_samples), device=tensor_args.device, dtype=tensor_args.dtype
        )
    return samples


def generate_gaussian_halton_samples(
    num_samples,
    ndims,
    bases=None,
    use_scipy_halton=True,
    seed=123,
    tensor_args=TensorDeviceType(),
    variance=1.0,
):
    uniform_halton_samples = generate_halton_samples(
        num_samples, ndims, bases, use_scipy_halton, seed, tensor_args=tensor_args
    )

    gaussian_halton_samples = torch.sqrt(
        torch.tensor([2.0], device=tensor_args.device, dtype=tensor_args.dtype)
    ) * torch.erfinv(2 * uniform_halton_samples - 1)

    # project them to covariance:
    i_mat = torch.eye(ndims, device=tensor_args.device, dtype=tensor_args.dtype)
    gaussian_halton_samples = torch.matmul(gaussian_halton_samples, np.sqrt(variance) * i_mat)
    return gaussian_halton_samples


def generate_gaussian_sobol_samples(
    num_samples,
    ndims,
    seed,
    tensor_args=TensorDeviceType(),
):
    soboleng = torch.quasirandom.SobolEngine(dimension=ndims, scramble=True, seed=seed)
    uniform_sobol_samples = soboleng.draw(num_samples).to(tensor_args.device)

    gaussian_sobol_samples = torch.sqrt(
        torch.tensor([2.0], device=tensor_args.device, dtype=tensor_args.dtype)
    ) * torch.erfinv(2 * uniform_sobol_samples - 1)
    return gaussian_sobol_samples
