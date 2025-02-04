import os
import re
import subprocess
import argparse


SHORT_Q = "short"
LONG_Q = "long"

VERBOSE = True
BSUB_COMMAND = "bsub -q short -R rusage[mem=1024] -C 1 -G fleishman-wx-grp-lsf -o %s -e %s.err -u /dev/null %s"
BASE_PATH = "./htfunclib/htFuncLib_notebook/bubb_design/"
ROSETTA_JOBS_FILE_NAME = "selected_combinations.txt"

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument("--bubb", type=str, help="Bubble index", required=False)
parser.add_argument("--njobs", type=int, help="Number of jobs to run", required=False)
parser.add_argument("--idx", type=int, help="Begining index to run within a bubble", required=False)
parser.add_argument("--mode", type=str, help="'scheduler' or 'worker'", required=True)

def read_file_line_by_line(filepath):
    with open(filepath, "r") as file:
        for line in file:
            yield line.strip()

def read_file(filepath):
    return(open(filepath, "r").readlines())

def scheduler_main():
    bubb_dirs = [bubb_dir for bubb_dir in os.listdir(BASE_PATH) if re.match(r'\d{1,3}A$', bubb_dir)]


def worker_main(bubb, idx, njobs):
    
    job_path = "%s/%s" % (BASE_PATH, bubb)
    rosetta_jobs = read_file("%s/%s" % (job_path, ROSETTA_JOBS_FILE_NAME))
    jobs_to_run = rosetta_jobs[idx:idx+njobs]

    if VERBOSE:
        print("Running for bubb %s, jobs [%d to %d]" % (bubb, idx, idx + njobs))
    

    outpath = "%s/output/%d_%d_track.txt"


    for job in jobs_to_run:
        
        
        print(job)
        process = subprocess.run(job, shell=True, check=True, text=True, capture_output=True)

        with open(outpath, 'a+') as track_file:
            track_file.write(job)
        



# /home/labs/fleishman/rosaliel/Rosetta/main/source/build/src/release/linux/3.10/64/x86/gcc/5.4/default/rosetta_scripts.default.linuxgccrelease -database /home/labs/fleishman/rosaliel/Rosetta/main/database @//home/labs/fleishman/itayta/htfunclib/htFuncLib_notebook/initial_data/flags -out:prefix 14AI_42AL_44AM_46AY_68AM_72AS_119AV_220AV_224AV_ -out:file:score_only //home/labs/fleishman/itayta/htfunclib/htFuncLib_notebook/bubb_design/42A/scores/14AI_42AL_44AM_46AY_68AM_72AS_119AV_220AV_224AV.sc -parser:script_vars all_ress=14A,16A,18A,42A,44A,46A,61A,64A,68A,69A,72A,108A,110A,112A,119A,123A,145A,150A,163A,165A,167A,181A,185A,201A,220A,224A new_res1=ILE target1=14A new_res2=LEU target2=42A new_res3=MET target3=44A new_res4=TYR target4=46A new_res5=MET target5=68A new_res6=SER target6=72A new_res7=VAL target7=119A new_res8=VAL target8=220A new_res9=VAL target9=224A 


args = parser.parse_args()

if args.mode == "scheduler":
    scheduler_main()
elif args.mode == "worker":
    worker_main(args.bubb, args.idx, args.njobs)
# Run the command

