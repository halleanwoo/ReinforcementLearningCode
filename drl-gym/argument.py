import argparse
import os
import time


parser = argparse.ArgumentParser()

parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4") # "Acrobot-v1"  "BreakoutNoFrameskip-v4"  CartPole-v0
parser.add_argument("--epoches", type=int, default=200, help="If flag_done_break is True, it is episodes; Else, it like periods")               # 100000
parser.add_argument("--max_step", type=int, default=5000, help="If flag_done_break, it usually has no means; Else, it work with epoches")           # 1000

parser.add_argument("--flag_done_break", type=bool, default=False, help="if True, when done,restart a new episode; Else, keep in this epoch")
parser.add_argument("--update_period", type=int, default=2500, help="period to update targetNet. If flag_done_break, it is episodes; else, steps")      # 2500
parser.add_argument("--reveal_period", type=int, default=5000, help="period to reveal the results, it only work when not flag_done_break")       # 2500
parser.add_argument("--test_period", type=int, default=10000, help="period to update targetNet. If flag_done_break, it is episodes; else, steps")

parser.add_argument("--memory_size", type=int, default=200000, help="If atari, suggesting lagrge; else, maybe 20000. Depend on memory of your computer")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate_Q", type=float, default=0.00025) #0.001

parser.add_argument("--seed", type=int, default=11037)
parser.add_argument("--env_seed", type=int, default=1234)

parser.add_argument("--observe_step", type=int, default=50000)         # 50000                 150
parser.add_argument("--explore_step", type=int, default=500000)     # 400000 1000000        10000
parser.add_argument("--train_step", type=int, default=3000)         # 

parser.add_argument("--frame_skip", type=int, default=4)

parser.add_argument("--flag_double_dqn", type=bool, default=False)
parser.add_argument("--flag_dueling_dqn", type=bool, default=False)
parser.add_argument("--flag_CLIP_NORM", type=bool, default=True)
parser.add_argument("--flag_flag_cnn", type=bool, default=True)

parser.add_argument('--file_path', type=str, default=str(os.getcwd()))
parser.add_argument('--flag_save', type=bool, default=True)
# file_path + file_path
parser.add_argument('--config_file', type=str, default='/config'+str(int(time.time()))+'.txt')
parser.add_argument('--train_file', type=str, default='/train_file'+str(int(time.time()))+'.csv')
parser.add_argument('--test_file', type=str, default='/test_file'+str(int(time.time()))+'.csv')
parser.add_argument('--all_file', type=str, default='/all_file'+str(int(time.time()))+'.csv')


args_origin = parser.parse_args()

def argsWrapper(args):
    experiment_name="seed1_dqn"
    # set config
    env_set = ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4"]
    if args.env_name in env_set:
        args.flag_cnn = True
    else:
        args.flag_cnn = False
    if not args.flag_cnn:
        print("make true your parms, such as memory_size, update_period")
        print(" ")
        print(" ")
    # set dir_path
    if args.flag_save:
        dir_path = str(int(time.time()))
        path = str(os.getcwd())
        new_path = path  + '/results/'  + experiment_name + "_" + args.env_name + dir_path
        # new_path = path  + '/' + args.env_name + dir_path
        os.makedirs(new_path)
        args.file_path = new_path
    args.config_file = args.file_path + args.config_file
    args.train_file = args.file_path + args.train_file
    args.test_file = args.file_path + args.test_file
    args.all_file = args.file_path + args.all_file
    return args

args = argsWrapper(args_origin)

print(args)