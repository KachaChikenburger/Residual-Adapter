import os
import sys
import time
import random
import argparse
import subprocess
import site

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

def get_dist_launch(args):  # some examples
    dist_presets = {
        'f4': ("0,1,2,3", 4, 9999),
        'f2': ("0,1", 2, 9999),
        'f3': ("0,1,2", 3, 9999),
        'f12': ("1,2", 2, 9999),
        'f02': ("0,2", 2, 9999),
        'f03': ("0,3", 2, 9999),
        'l2': ("2,3", 2, 9998),
    }

    if args.dist in dist_presets:
        visible_devices, world_size, master_port = dist_presets[args.dist]
    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        visible_devices, world_size, master_port = str(num), 1, 9999
    else:
        raise ValueError

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["WORLD_SIZE"] = str(world_size)

    lib_paths = []
    conda_lib = os.path.join(sys.prefix, "lib")
    if os.path.isdir(conda_lib):
        lib_paths.append(conda_lib)

    for site_dir in site.getsitepackages():
        torch_lib = os.path.join(site_dir, "torch", "lib")
        if os.path.isdir(torch_lib):
            lib_paths.append(torch_lib)

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        lib_paths.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

    launch_cmd = [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        "torch.distributed.launch",
        "--master_port",
        str(master_port),
        "--nproc_per_node",
        str(world_size),
        "--nnodes=1",
    ]
    return launch_cmd, env


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_retrieval(args):
    launch_cmd, env = get_dist_launch(args)
    cmd = launch_cmd + [
        "--use_env",
        "Retrieval.py",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--bs",
        str(args.bs),
        "--checkpoint",
        args.checkpoint
    ]
    if args.evaluate:
        cmd.append("--evaluate")

    print(f"Launching retrieval with interpreter: {sys.executable}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def run(args):
    if args.task == 'itr_rsicd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_vit.yaml'
        run_retrieval(args)

    elif args.task == 'itr_rsitmd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_vit.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsitmd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_geo.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsicd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_geo.yaml'
        run_retrieval(args)

    elif args.task == 'itr_coco':
        assert os.path.exists("../X-VLM-pytorch/images/coco")
        args.config = 'configs/Retrieval_coco.yaml'
        run_retrieval(args)

    elif args.task == 'itr_nwpu':
        assert os.path.exists("../X-VLM-pytorch/images/NWPU")
        args.config = 'configs/Retrieval_nwpu.yaml'
        run_retrieval(args)
    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='itr_rsitmd')
    parser.add_argument('--dist', type=str, default='f2', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsitmd_vit.yaml', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                  "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./outputs/test', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    args = parser.parse_args()
    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)
    run(args)
