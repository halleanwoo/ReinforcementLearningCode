import argparse
import os
import time

experiment_name="XXX"

parser = argparse.ArgumentParser()

parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4") # "Acrobot-v1"  "BreakoutNoFrameskip-v4"  CartPole-v0
parser.add_argument("--epoches", type=int, default=10)
parser.add_argument("--max_step", type=int, default=100)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--memory_size", type=int, default=200000)
parser.add_argument("--learning_rate_Q", type=float, default=0.00025) #0.001

parser.add_argument("--seed", type=int, default=11037)

parser.add_argument("--observe_step", type=int, default=50000)      # 50000
parser.add_argument("--explore_step", type=int, default=1000000)
parser.add_argument("--train_step", type=int, default=3000)

parser.add_argument("--update_period", type=int, default=2500)      # 2500
parser.add_argument("--reveal_period", type=int, default=100)       # 2500
parser.add_argument("--test_period", type=int, default=2)
parser.add_argument("--frame_skip", type=int, default=4)

parser.add_argument("--flag_double_dqn", type=bool, default=False)
parser.add_argument("--flag_dueling_dqn", type=bool, default=False)
parser.add_argument("--flag_CLIP_NORM", type=bool, default=True)
parser.add_argument("--flag_flag_cnn", type=bool, default=True)

parser.add_argument("--flag_done_break", type=bool, default=True)

parser.add_argument('--file_path', type=str, default='zResult/')
parser.add_argument('--flag_save', type=bool, default=False)
# file_path + file_path
parser.add_argument('--train_file', type=str, default=experiment_name + '/train_file.csv')
parser.add_argument('--test_file', type=str, default=experiment_name + '/test_file.csv')
parser.add_argument('--all_file', type=str, default=experiment_name + '/all_file.csv')


args_origin = parser.parse_args()

def argsWrapper(args):
    # set dir_path
    if args.flag_save:
        dir_path = str(int(time.time()))
        path = str(os.getcwd())
        new_path = path  + '/' + args.env_name + dir_path + "/"
        os.makedirs(new_path)
        args.file_path = new_path
    args.train_file = args.file_path + args.train_file
    return args

args = argsWrapper(args_origin)