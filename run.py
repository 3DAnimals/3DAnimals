import hydra
from omegaconf import DictConfig
from model import Trainer, build_model

# tensorboard --logdir ./tensorboard_logs/ --bind_all --samples_per_plugin images=1000000000 --port=23425 --reload_multifile True --load_fast false

@hydra.main(config_path="config", config_name="train_magicpony_horse")
def main(cfg: DictConfig):
    model = build_model(cfg.model)
    trainer = Trainer(cfg, model)

    if cfg.run_train:
        trainer.train()
    if cfg.run_test:
        trainer.test()


if __name__ == "__main__":
    main()
