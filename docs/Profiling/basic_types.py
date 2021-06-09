import numpy as np
from tardis.montecarlo.montecarlo_numba.r_packet import (RPacket, trace_packet)
from tardis.montecarlo.montecarlo_numba.numba_interface import (NumbaModel, 
                                                                NumbaPlasma, 
                                                                numba_plasma_initialize, 
                                                                Estimators)

def get_numba_model(sim):
    runner = sim.runner
    model = sim.model
    return NumbaModel(
        runner.r_inner_cgs,
        runner.r_outer_cgs,
        model.time_explosion.to("s").value,
    )

def get_r_packet(**kwargs):
    r_packet_kwargs = dict(
        r = 7.5e14,
        mu = 0.3,
        nu = 0.4,
        energy = 0.9,
        seed = 1963,
        index = 0,
        is_close_line = 0,
    )
    r_packet_kwargs.update(kwargs)
    return RPacket(**r_packet_kwargs)

def get_numba_plasma(sim):
    return numba_plasma_initialize(sim.plasma, sim.runner.line_interaction_type)

def get_estimator(sim):
    runner = sim.runner
    return Estimators(
        runner.j_estimator,
        runner.nu_bar_estimator,
        runner.j_blue_estimator, 
        runner.Edotlu_estimator,
    )
