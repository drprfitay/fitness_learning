
import subprocess
import os
import random


def run_python_script(args, cwd=None, return_both=False):
    """
    Executes a Python script with the given arguments list in the specified directory.
    Prints STDOUT in real time unless return_both=True, in which case returns (stdout, stderr) as strings.
    The first element should be the script path, followed by its arguments.
    Example: ["other_script.py", "--foo", "bar"]
    """
    cmd = ["python"] + args
    print("Running command:", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    captured_stdout = ""
    captured_stderr = ""

    while True:
        stdout_line = process.stdout.readline()
        if stdout_line:
            if return_both:
                captured_stdout += stdout_line
            else:
                print(stdout_line, end="", flush=True)
        elif process.poll() is not None:
            break

    # Drain any remaining stdout lines
    for line in process.stdout:
        if return_both:
            captured_stdout += line
        else:
            print(line, end="", flush=True)

    # Drain stderr
    for line in process.stderr:
        if return_both:
            captured_stderr += line
        else:
            captured_stderr += line  # accumulate to print below

    if captured_stderr and not return_both:
        print("\n--- STDERR ---\n", captured_stderr)

    process.stdout.close()
    process.stderr.close()

    if return_both:
        return (captured_stdout, captured_stderr)
    return process

    
def run_local_job(python_command):
    """
    Runs a python command (as a string) using subprocess in the local shell.

    Args:
        python_command (str): The full python command to execute, e.g., "python script.py --arg1 val1"
    """
    try:
        print(f"Running local job: {python_command}")
        result = subprocess.run(python_command, shell=True, check=True, capture_output=True, text=True)
        print("Job output:\n", result.stdout)
        if result.stderr:
            print("Job error output:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running local job: {e}\nOutput: {e.output}\nStderr: {e.stderr}")



def run_lsf_job(python_run_command, 
                local_script_dir, 
                code_execution_dir, 
                conda_env_name, 
                worker_idx=None,
                gpu=True,
                N_cores=2,
                rusage="64GB"):

    os.makedirs(local_script_dir, exist_ok=True)
    job_id = f"worker_{random.randint(10000000, 99999999)}"
    bash_script_path = os.path.join(local_script_dir, f"job_{job_id}.sh")
    output_dir = os.path.join((local_script_dir), "job_outputs")
    os.makedirs(output_dir, exist_ok=True)
    err_file = os.path.join(output_dir, f"err_file_{job_id}")
    out_file = os.path.join(output_dir, f"out_file_{job_id}")
 

    bash_script_lines = [
        "#!/bin/bash",
        "source ~/.bashrc",
        "cd %s" % code_execution_dir,
        "conda activate %s" % conda_env_name,
        python_run_command
    ]

    with open(bash_script_path, "w") as f:
        f.write('\n'.join(bash_script_lines) + '\n')

    subprocess.check_call(['chmod', '+x', bash_script_path])
    
    if gpu:
        bsub_cmd = [
            'bsub',
            '-n', '%d' % N_cores,
            '-gpu', 'num=1:gmem=24G:aff=yes',
            '-R', 'same[gpumodel]',
            '-R', 'rusage[mem=%s]' % rusage,
            '-R', 'span[ptile=%d]' % N_cores,
            '-e', err_file,
            '-o', out_file,
            '-q', 'short-gpu',
            bash_script_path
        ]

    else:
        bsub_cmd = [
            'bsub',
            '-n', '%d' % N_cores,
            '-R', 'rusage[mem=%s]' % rusage,
            '-R', 'span[ptile=%d]' % N_cores,
            '-e', err_file,
            '-o', out_file,
            '-q', 'short',
            bash_script_path
        ]        
    
    print(f"Submitting job {worker_idx+1 if worker_idx is not None else ''} via: ", " ".join(bsub_cmd))
    try:
        result = subprocess.run(bsub_cmd, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Job submission failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
        else:
            print(f"Job submitted successfully. Output: {result.stdout}")
    except FileNotFoundError as e:
        print(f"Error: 'bsub' command not found. LSF may not be installed or not in PATH.")
        print(f"Details: {e}")
        print(f"Script created at: {bash_script_path} (but job was not submitted)")
    except Exception as e:
        print(f"Unexpected error during job submission: {e}")
        print(f"Script created at: {bash_script_path} (but job was not submitted)")

    return (bash_script_path, err_file, out_file)
    # Optionally delete temp script afterwards (uncomment if desired)
    # import time; time.sleep(2)
    # os.remove(bash_script_path)