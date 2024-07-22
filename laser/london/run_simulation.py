"""
"""

from pathlib import Path
import numpy as np
import os

from idmlaser.utils import PropertySet
from analyze import main as analyze_wave  # noqa: E402, I001

def log_prob(x):
    mu = np.array([-1.10, -3.6])
    cov = np.diag([0.15, 2.9])**2
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def main(params={}, do_plot=False):
    # change directory to the parent of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    meta_params = PropertySet()
    meta_params.ticks = 365 * 5
    meta_params.nodes = 0
    meta_params.seed = 20240612
    meta_params.output = Path.cwd() / "outputs"

    model_params = PropertySet()
    model_params.exp_mean = np.float32(7.0)
    model_params.exp_std = np.float32(1.0)
    model_params.inf_mean = np.float32(7.0)
    model_params.inf_std = np.float32(1.0)
    model_params.r_naught = np.float32(14.0)
    model_params.seasonality_factor = np.float32(0.1)
    model_params.seasonality_offset = np.int32(182.5)

    model_params.beta = model_params.r_naught / model_params.inf_mean

    # London Scenario
    meta_params.scenario = "london"

    # England and Wales network parameters (we derive connectivity from these and distance)
    net_params = PropertySet()
    net_params.a = np.float32(1.0)   # pop1 power
    net_params.b = np.float32(1.0)   # pop2 power
    net_params.c = np.float32(params.get("c",2.0))   # distance power
    net_params.k = np.float32(params.get("k",500.0)) # scaling factor
    net_params.max_frac = np.float32(0.05) # max fraction of population that can migrate

    from scenario_london import initialize  # noqa: E402, I001
    params = PropertySet(meta_params, model_params, net_params)
    max_capacity, demographics, initial, network = initialize(None, params, params.nodes)    # doesn't need a model, yet

    from datetime import datetime

    params.prng_seed = datetime.now(tz=None).microsecond  # noqa: DTZ005

    # CPU based implementation
    from idmlaser.models.numpynumba import NumbaSpatialSEIR  # noqa: I001, E402, RUF100
    model = NumbaSpatialSEIR(params)

    # # GPU based implementation with Taichi
    # from idmlaser.models.taichi import TaichiSpatialSEIR  # noqa: I001, E402, RUF100
    # model = TaichiSpatialSEIR(params)

    model.initialize(max_capacity, demographics, initial, network)

    model.run(params.ticks)

    # analyze phase
    p = analyze_wave(model.report, do_plot=do_plot)

    # return log_prob (maximize this)
    return log_prob(p)

if __name__ == "__main__":
    main()

    # paramfile, npyfile = model.finalize(prefix="")

