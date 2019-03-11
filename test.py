import torch
from trainer import Trainer, parse_config

def test(config):
    trainer = Trainer(config)
    trainer.test_model()

if __name__ == "__main__":
    config = parse_config()
    test(config)

