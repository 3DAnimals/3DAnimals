from .Trainer import Trainer
from .models import MagicPony
from .models import Ponymation
from .models import FaunaModel

def build_model(cfg):
    if cfg.name == "MagicPony":
        return MagicPony(cfg)
    elif cfg.name == "Ponymation":
        return Ponymation(cfg)
    elif cfg.name == "Fauna":
        return FaunaModel(cfg)
    else:
        raise NotImplementedError(f"Unrecognized name in model cfg: {cfg.name}")
