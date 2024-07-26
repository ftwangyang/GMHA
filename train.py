import os
import argparse
from datetime import datetime
import pytz
import pprint as pp
import torch

from Trainer import Trainer
from utils import seed_everything, occumpy_mem


def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size}
    model_params = {
        "embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
        "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
        "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim, "num_experts": args.num_experts, "eval_type": args.eval_type,
        "norm": args.norm, "norm_loc": args.norm_loc, "expert_loc": args.expert_loc, "problem": args.problem,
        "topk": args.topk, "routing_level": args.routing_level, "routing_method": args.routing_method
    }
    optimizer_params = {
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}
    }
    trainer_params = {
        "epochs": args.epochs, "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size, "validation_interval": args.validation_interval,
        "entropy_weight" : args.entropy_weight,
        "model_save_interval": args.model_save_interval, "checkpoint": args.checkpoint
    }
    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maximum Entropy Reinforcement Learning with Gated Multi-Head Attention for Cross-Problem Generalization in Vehicle Routing Problems")

    # env_params
    parser.add_argument('--problem', type=str, default="Train_ALL", choices=[
        "Train_ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
        "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
        "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
    ])
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--pomo_size', type=int, default=100)

    # model_params
    parser.add_argument('--model_type', type=str, default="GMHA*", choices=["GMHA", "GMHA*"])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128 ** 0.5)
    parser.add_argument('--encoder_layer_num', type=int, default=6)
    parser.add_argument('--decoder_layer_num', type=int, default=1)
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"])
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--expert_loc', type=str, nargs='+', default=[
        'Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'
    ])
    parser.add_argument('--routing_level', type=str, default="node", choices=["node", "instance", "problem"])
    parser.add_argument('--routing_method', type=str, default="input_choice", choices=["input_choice", "expert_choice", "soft_moe", "random"])

    # optimizer_params
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=[4501])
    parser.add_argument('--gamma', type=float, default=0.1)

    # trainer_params
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--train_episodes', type=int, default=20000)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--entropy_weight', type=float, default=0.01)
    parser.add_argument('--validation_interval', type=int, default=50)
    parser.add_argument('--model_save_interval', type=int, default=2500)
    parser.add_argument('--checkpoint', type=str, default=None)

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--occ_gpu', type=float, default=0.0)

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    seed_everything(args.seed)

    # set log & gpu
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    args.log_path = os.path.join(args.log_dir, process_start_time.strftime("%Y%m%d_%H%M%S"))
    print(">> Log Path: {}".format(args.log_path))
    os.makedirs(args.log_path, exist_ok=True)
    if not args.no_cuda and torch.cuda.is_available():
        occumpy_mem(args) if args.occ_gpu != 0.0 else print(">> No occupation needed")
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    # start training
    print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    trainer = Trainer(args=args, env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    trainer.run()
    print(">> Finish {} Training using {} Model ...".format(args.problem, args.model_type))
