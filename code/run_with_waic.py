import argparse
import builtins
import datetime
import os
import random
import shutil
import string
import subprocess
from pathlib import Path


def restart_from_checkpoint_using_fabric(fabric, ckp_path, state_to_load, run_variables=None):
    """
    Re-start from checkpoint
    """
    try:
        reminder = fabric.load(ckp_path, state_to_load)
    except FileNotFoundError:
        print(f'checkoint {ckp_path} does not exist. skipping.')
        return
    print(f"Found checkpoint at {ckp_path}")

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in reminder:
                run_variables[var_name] = reminder[var_name]


def save_using_fabric(fabric, args, epoch, model, optimizer, **kwargs):
    to_save = {
        'model': model,
        'optimizer': optimizer,
        'grad_scaler': fabric.strategy.precision,  # does not really matter.
        'args': args,
        'epoch': epoch
    }
    to_save.update(kwargs)
    checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
    candidate = os.path.join(args.output_dir, f'checkpoint-{epoch:05d}.pth')
    if epoch >= args.epochs - 1:
        checkpoint_paths.append(candidate)
    elif epoch > 0 and (epoch % getattr(args, 'save_every', args.epochs) == 0):
        checkpoint_paths.append(candidate)
    for checkpoint_path in checkpoint_paths:
        fabric.save(checkpoint_path, to_save)


def lsf_kill_this_seq_array_jobs():
    # USE WIITH CARE!
    # run this _inside_ a job - it will kill it and all subsequent jobs in the same job array.
    jobid = os.environ.get('LSB_JOBID')
    cmd = f'bkill "{jobid}"'
    subprocess.call(cmd, shell=True)


def lsf_get_time_left_for_job():
    # run this _inside_ the job to know how long it has until termination
    # get jobid + array (https://www.ibm.com/support/pages/accessing-lsf-batch-job-id-and-array-id-within-job-environment)
    jobid = os.environ.get('LSB_BATCH_JID')
    cmd = f'bjobs -noheader -o "time_left" "{jobid}"'
    sec = None
    try:
        out = subprocess.check_output(cmd, shell=True)
        t = out.decode().strip().split(' ')[0]
        sec = 0
        cur = 60
        for i in reversed(t.split(':')):
            sec += cur * int(i)
            cur *= 60
        print(f'\t-lsf time remain- Got {out.decode().strip()} = {sec} [sec]')
    except Exception as e:
        print(f'Cannot get remaining time for job {jobid}: {e} ({type(e).__name__})')
    return sec


def lsf_get_num_devices_and_num_nodes_for_current_job():
    # run _inside_ job to know how many NODES are being utilized
    hosts = {}
    for h in os.environ['LSB_HOSTS'].split(' '):
        hosts[h] = hosts.get(h, 0) + 1
    cur_host = os.uname().nodename.split('.')[0]
    # print(f'{os.environ["LSB_HOSTS"]} -> {hosts}. current host = {cur_host}')
    num_nodes = len(hosts.keys())
    num_devices = hosts[cur_host]
    return num_devices, num_nodes


def lsf_get_all_queues_for_user():
    """
    get all the possible queues the user can run on
    :return: string
    """
    return "waic-long waic-short waic-risk long-gpu short-gpu risk-gpu"
    q = subprocess.run(f'bqueues -u {os.environ["USER"]}  -o "QUEUE_NAME" -noheader',
                       shell=True, capture_output=True, text=True)
    return q.stdout.replace('\n', ' ').strip()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def lsf_write_job_script(output_dir, job_name, logpath, ptile, args, script_cmd):
    filename = os.path.join(output_dir, 'bsub_cmd.sh')
    with open(filename, 'w') as R:
        R.write('#!/bin/bash\n')
        if args.cont > 1:
            R.write('#BSUB -H\n')
            R.write(f'#BSUB -J "{job_name}[1-{args.cont}]"\n')
        else:
            R.write(f'#BSUB -J "{job_name}"\n')
        R.write(f'#BSUB -q "{args.waic_q}"\n')
        R.write(f'#BSUB -n {args.ngpus}\n')
        R.write(f'#BSUB -gpu num=1:j_exclusive=yes:gmem={args.gpu_mem}G:aff=yes\n')  # :mps=yes,shared,nocvd\n')
        R.write(f'#BSUB -R rusage[mem=48000]\n')
        # R.write(f'#BSUB -R affinity[thread*10,distribute=pack]\n')
        R.write(f'#BSUB -R affinity[core(5)]\n')
        R.write(f'#BSUB -R same[gpumodel]\n')
        R.write(f'#BSUB -R span[ptile={ptile}]\n')  # once fabric bug is fixed, this can be removed
        R.write(f'#BSUB -o {logpath}.o\n')
        R.write(f'#BSUB -e {logpath}.e\n')
        if len(args.bsub_args.strip()) > 0:
            R.write(f'#BSUB {args.bsub_args}\n')
        R.write('module purge\n')
        R.write('ml Miniconda3\n')
        R.write('ml local\n')
        R.write('source deactivate\n')
        if os.path.isdir(args.conda_env):
            R.write(f'source activate {os.path.abspath(args.conda_env)}\n')
        else:
            R.write(f'source activate {os.environ["HOME"]}/.conda/envs/{args.conda_env}\n')
        R.write('nvidia-smi\n')
        R.write(f'cd {output_dir}\n')
        R.write(f'mpiexec -rmk lsf -prepend-rank -profile python -u {script_cmd} '
                f'--precision={args.precision} --accelerator={args.accelerator} --strategy={args.strategy}\n')
    return filename


def _mirror_project_to_subfolder(src, dst):
    ext_to_copy = ('py', 'json', 'yaml')

    for filename in os.listdir(src):
        # ignore hidden files / folders
        if filename.startswith('.'):
            continue

        cur = os.path.join(src, filename)
        if os.path.isdir(cur):
            # recurse into sub modules
            # Copy only python packages (e.g. directories that contain '__init__.py' file)
            if '__init__.py' in os.listdir(cur):
                # recursive call
                new_dst = os.path.join(dst, filename)
                os.makedirs(new_dst, exist_ok=True)
                _mirror_project_to_subfolder(cur, new_dst)
        else:
            # regular files
            ext = '.'.join(filename.split('.')[1:]).lower()

            if ext in ext_to_copy:
                # file to copy
                shutil.copy(cur, dst)
            else:
                # files to symbolic links
                os.symlink(src=os.path.abspath(cur), dst=os.path.join(dst, filename))


def startup(args):
    # find available working dir
    v = 0
    while True:
        output_dir = os.path.abspath(os.path.join(args.working_dir_base, f'{args.tag}-v{v}'))
        if not os.path.isdir(output_dir):
            break
        v += 1
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f'-startup- working directory is {output_dir}')

    # copy files. symbolic link weight files
    _mirror_project_to_subfolder('.', output_dir)

    return output_dir


def mirror_and_submit():
    parser = argparse.ArgumentParser('WAIC mirror and submit')
    parser.add_argument('--tag', default='name_of_this_experiment', type=str, help='tag for this run - experiment description')
    parser.add_argument('--working_dir_base', default='./experiments', type=str, help='master folder for all experiments')
    parser.add_argument('--cont', default=20, type=int, help='number of jobs in seq array')
    parser.add_argument('--ngpus', default=8, type=int, help='number of GPUs')
    parser.add_argument('--ptile', default=8, type=int, help='number of GPUs on a SINGLE node')
    parser.add_argument('--gpu_mem', default=20, type=int, help='minimal GPU memory in GB')
    parser.add_argument('--waic_q', default=None, type=str, help='LSF queue name')
    parser.add_argument('--bsub_args', default='', type=str, help='additional bsub arguments')
    parser.add_argument('--conda_env', type=str, help='name of conda environment to use')

    # arguments for Fabric -- better use the defaults
    # see https://lightning.ai/docs/fabric/stable/fundamentals/launch.html for details
    parser.add_argument('--accelerator', default='cuda', type=str, choices=('cuda', 'cpu'))
    parser.add_argument('--strategy', default='ddp', type=str, choices=('ddp', 'fsdp', 'dp', ))
    parser.add_argument('--precision', default='16-mixed', type=str,
                        choices=('16-mixed', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'))

    # wandb arguments -- only if you EXPLICILTY know them
    parser.add_argument('--wandb_project', default=None, type=str, help='name of wandb project')
    parser.add_argument('--wandb_id', default=None, type=str, help='run id for wandb')

    # do not mess with this
    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')

    args, main_script = parser.parse_known_args()

    # create new folder for this run
    if args.output_dir is None:
        output_dir = startup(args)
        job_name = os.path.basename(output_dir)
        if len(job_name) == 0:
            job_name = os.path.basename(output_dir[:-1])
        log = os.path.join(output_dir, 'log')
    else:
        output_dir = args.output_dir
        job_name = 'log_' + ''.join(random.choice(string.digits + string.ascii_letters) for _ in range(10))
        log = os.path.join(output_dir, job_name)
    main_script.extend(('--output_dir', output_dir))

    # handle wandb arguments
    if args.wandb_project is None:
        proj_name = os.path.basename(os.path.abspath(args.working_dir_base))
        if len(proj_name) == 0:
            proj_name = os.path.basename(os.path.abspath(args.working_dir_base[:-1]))
        args.wandb_project = proj_name
        args.wandb_id = job_name
    main_script.extend(
        (
            '--wandb_project', args.wandb_project,
            '--wandb_id', args.wandb_id,
        )
    )

    script_cmd = ' '.join(main_script)

    logpath = os.path.join(os.path.dirname(log), 'log')

    # Build the command to run
    # ---------------------------------------------------------------------
    #
    # Need to change here for your modules/environment/conda
    #
    # ---------------------------------------------------------------------
    # select queues
    if args.waic_q is None:
        args.waic_q = lsf_get_all_queues_for_user()

    # tiles and tasks (when bug is fixed in fabric - this can be removed)
    num_tasks = args.ngpus
    if num_tasks > args.ptile:
        # multi node
        ptile = args.ptile
        num_nodes = num_tasks // ptile
        assert num_nodes * ptile == num_tasks, f'Fill the nodes! if ngpus {args.ngpus} > {ptile}, it must be an int mult of {ptile}'
    else:
        ptile = args.ngpus

    bsub_script = lsf_write_job_script(output_dir=output_dir, job_name=job_name, logpath=logpath,
                                       ptile=ptile, args=args, script_cmd=script_cmd)
    if args.cont > 1:
        # submit an array
        subprocess.call(f'/home/projects/bagon/shared/seq_arr.sh -e {args.cont:d} -f {bsub_script}', shell=True)
    else:
        # submit a single job
        print(f'Submitting {bsub_script}...')
        subprocess.call(f'bsub < {bsub_script}', shell=True)


if __name__ == '__main__':
    mirror_and_submit()
