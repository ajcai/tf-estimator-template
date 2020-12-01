import argparse
from trainer import train, evaluate, predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', help='train, eval or predict')
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    if args.mode == "eval":
        evaluate()
    if args.mode == "predict":
        predict()
        
    
