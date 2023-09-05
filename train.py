from cifar10.certify import main as certify_main
import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from common import get_args


def main_spawn(args):
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.node_num
    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


def main_worker(gpu, args):
    args.global_rank = gpu
    args.local_rank = gpu % args.ngpus_per_node
    torch.cuda.set_device(args.local_rank)
    print('global_rank ', args.global_rank, ' local_rank ', args.local_rank, ' GPU ', torch.cuda.current_device())
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.global_rank)
    certify_main(args)

def multinode_start(args, env_args):
    args.master_uri = "tcp://%s:%s" % (env_args.get("MASTER_ADDR"), env_args.get("MASTER_PORT"))
    args.node_rank = env_args.get("NODE_RANK")
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = env_args.get("WORLD_SIZE")
    args.local_rank = env_args.get("LOCAL_RANK")
    args.global_rank = env_args.get("RANK")
    torch.cuda.set_device(args.local_rank)
    print('global_rank ', args.global_rank, ' local_rank ', args.local_rank, ' GPU ', torch.cuda.current_device())
    dist.init_process_group(backend=args.dist_backend, init_method=args.master_uri, world_size=args.world_size, rank=args.global_rank)
    certify_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, default=0.25, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--outfile", type=str, default='certify_', help="output file")
    parser.add_argument("--dataset", type=str, default='cifar10', help="dataset name")
    parser.add_argument("--node_num", type=int, default=1, help="node number")
    parser.add_argument("--dist_backend", type=str, default='nccl', help="distributed backend")
    parser.add_argument("--dist_url", type=str, default='tcp://127.0.0.1:23456', help="distributed url")
    parser.add_argument("--workers", type=int, default=2, help="number of workers")
    args = parser.parse_args()

    args.output = os.environ.get('AMLT_OUTPUT_DIR', os.path.join('/D_data/kaqiu/diffusion_denoise_smoothing/', args.dataset))
    args.data = os.environ.get('AMLT_DATA_DIR', os.path.join('/D_data/kaqiu', args.dataset))
    if '/D_data/kaqiu' in args.data: # local
        args.local = 1
        args.smoothing_path = '../amlt'
    else: # itp
        args.local = 0
        args.smoothing_path = args.data

    args.retry_path = os.path.join(args.data, 'diffusion_denoise_smoothing/reproduce')
    args.outdir = os.path.join(args.output, 'reproduce')
    args.outfile = os.path.join(args.outdir, args.outfile + str(args.sigma))

    if args.node_num > 1:
        env_args = get_args()
        if env_args.get('RANK') == 0 and not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if env_args.get('RANK') == 0 and not os.path.exists(args.retry_path):
            os.makedirs(args.retry_path)
        multinode_start(args, env_args)
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if not os.path.exists(args.retry_path):
            os.makedirs(args.retry_path)

        main_spawn(args)
