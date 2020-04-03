import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run_block(rank, size):
    """Distributed function """
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        #send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        #Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data', tensor[0])

def run_non_block(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        #send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        #Receive tensor fom process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, 'has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """Initialize the distibuted environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    print('init process')
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_non_block))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
