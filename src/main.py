from data_preparation import data_prep
from data_preprocessing import data_preprocess
from feature_engineering import feature_engineering
from modelling import modelling

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        type= str,
        choices= [
            'run_with_training',
            'run_without_training'
        ]
    )

    args = parser.parse_args()

    if args.command == 'run_with_training':
        data_prep()
        data_preprocess()
        feature_engineering()
        modelling()
    elif args.command == 'run_without_training':
        print('test')
    else:
        raise RuntimeError('Invalid Argument!')

if __name__ == '__main__':
    main()

