import argparse
import json

from runners import runners


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = json.loads(open(args.config).read())
    runner_type = getattr(runners, config['type'])

    print("Trainer: ", config['type'], runner_type)
    runner = runner_type(config)
    runner.inference()
