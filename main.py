import argparse
import json

from trainers import trainers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = json.loads(open(args.config).read())
    trainer_type = getattr(trainers, config['type'])

    print("Trainer: ", config['type'], trainer_type)
    trainer = trainer_type(config)
    trainer.loop()
