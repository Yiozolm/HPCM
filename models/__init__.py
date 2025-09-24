from compressai.models import (
    FactorizedPrior,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
    Cheng2020Anchor,
    Cheng2020Attention,
)

from .HPCM_Base import HPCM_Base
from .HPCM_Large import HPCM_Large

image_models = {
    "factorized": FactorizedPrior,
    "hyperprior": ScaleHyperprior,
    "mbt2018-mean": MeanScaleHyperprior,
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
    "cheng2020-attn": Cheng2020Attention,
}


models = {}
models.update(image_models)

models["hpcm-base"] = HPCM_Base
models["hpcm-large"] = HPCM_Large