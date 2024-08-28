from dataclasses import dataclass
from .AnimalModel import AnimalModel, AnimalModelConfig
from ..utils import misc
from ..predictors import BasePredictorBase, InstancePredictorBase, BasePredictorConfig, InstancePredictorConfig


@dataclass
class MagicPonyConfig(AnimalModelConfig):
    cfg_predictor_base: BasePredictorConfig = None
    cfg_predictor_instance: InstancePredictorConfig = None


class MagicPony(AnimalModel):
    def __init__(self, cfg: MagicPonyConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, MagicPonyConfig)
        self.netBase = BasePredictorBase(self.cfg_predictor_base)
        self.netInstance = InstancePredictorBase(self.cfg_predictor_instance)
