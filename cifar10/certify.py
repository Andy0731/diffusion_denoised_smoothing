import os 
import argparse
import time 
import datetime 
from torchvision import transforms, datasets
import torch
import torch.distributed as dist

from .core import Smooth 
from .DRM import DiffusionRobustModel


def merge_ctf_files(ctf_filename, args):
    with open(ctf_filename, 'w') as outfile:
        for i in range(args.world_size):
            fname = ctf_filename + '_rank' + str(i)
            with open(fname) as infile:
                outfile.write(infile.read())
    return


def main(args):
    model = DiffusionRobustModel(args.data)

    dataset = datasets.CIFAR10(args.data, train=False, download=False, transform=transforms.ToTensor())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, pin_memory=True, sampler=sampler)

    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier 
    smoothed_classifier = Smooth(model, 10, args.sigma, t)

    print('begin to certify test set ...')
    args.outfile = args.outfile + '_rank' + str(args.global_rank)

    f = open(args.outfile, 'w')
    if args.global_rank == 0:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    for i, (x, label) in enumerate(loader):
        if i % args.skip != 0:
            continue

        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()

        correct += int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        total_num += 1

        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i * args.world_size + args.global_rank, label, prediction, radius, correct, time_elapsed))        

        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i * args.world_size + args.global_rank, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    print("GPU %d sigma %.2f accuracy of smoothed classifier %.4f "%(args.global_rank, args.sigma, correct/float(total_num)))
    close_flag = torch.ones(1).cuda()
    print('rank ', args.global_rank, ', close_flag ', close_flag)
    dist.all_reduce(close_flag, op=dist.ReduceOp.SUM)
    print('rank ', args.global_rank, ', close_flag ', close_flag)
    if args.global_rank == 0:
        ctf_filename = args.outfile.replace('_rank0', '')
        merge_ctf_files(ctf_filename, args)    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Predict on many examples')
    # parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    # parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    # parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    # parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    # parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    # parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    # parser.add_argument("--outfile", type=str, help="output file")
    # args = parser.parse_args()

    main(args)