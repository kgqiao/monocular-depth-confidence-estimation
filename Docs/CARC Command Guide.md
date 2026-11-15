

Reference:

> https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/slurm-cheatsheet


## Nodes

```
ssh user@10.72.0.13
```


| Node                 | Command                                         | Use                                             |
| -------------------- | ----------------------------------------------- | ----------------------------------------------- |
| Discovery login 1    | `ssh $USER@10.72.0.13`                          | Submit jobs, edit files, short checks           |
| Discovery login 2    | `ssh $USER@10.72.0.14`                          | Same as login 1                                 |
| Endeavour login 1, 2 | `ssh $USER@10.72.0.15`, `ssh $USER@10.72.0.16`  | Condo nodes of research groups only; not needed |
| Transfer node 1      | `ssh $USER@hpc-transfer1.usc.edu` (68.181.11.5) | Long, high-volume downloads                     |
| Transfer node 2      | `ssh $USER@hpc-transfer2.usc.edu` (68.181.11.4) | Same as transfer node 1                         |

- Discovery and Endeavour share `/home1`, `/project` and `/scratch1`.

## Accounts and quotas

```shell
myaccount                # Slurm accounts, service units, partitions
myquota                  # bytes and file counts on /home1, /project, /scratch1
sinfo -s                 # partitions of this cluster
id                       # my Unix groups
sacctmgr show assoc user=$USER format=account%30,partition%20,qos%30   # if myaccount is missing
```

Always pass `--account=yzhao010_1531` (or `-A yzhao010_1531`) to `srun` and `sbatch`. 


```
echo 'export SBATCH_ACCOUNT=yzhao010_1531' >> ~/.bashrc
```

```
source ~/.bashrc
```


## Storage

| Path | Use |
|---|---|
| `/scratch1/$USER` | Large data, such as the full nuScenes. Only I can read it by default. No backup. |
| `/project/<pi>_<id>` | Shared by the project group, but small |

To let the project group read a folder in my scratch:

```shell
G=<pi>_<id>                                    # the project group from `id`
chmod g+x /scratch1/$USER
chgrp -R $G /scratch1/$USER/nuscenes
chmod -R g+rX /scratch1/$USER/nuscenes         # g+rwX to let them write
chmod g+s /scratch1/$USER/nuscenes             # new files keep the group
```

## Copy files from the laptop

From the repository root on the laptop:

```shell
scp -r utilities $USER@10.72.0.13:~/
rsync -av --exclude __pycache__ utilities $USER@10.72.0.13:~/     # same, without __pycache__
```

## Set up uv

The compute nodes see my home folder, so one install serves the jobs too.

```shell
which uv                                           # empty: not installed
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs to ~/.local/bin
source ~/.local/bin/env
uv --version
```

If uv cannot install, use a module Python 3.10 or later. The system `python3` is too old.

```shell
module spider python
module load python/<version>
```

## Slurm jobs


### Single Command

```shell
srun --account=<pi> --partition=main --time=00:20:00 <command>   
```

### Batch Job

```shell
sbatch --account=<pi> <script>.sbatch   
```

### Bash Session

```shell
srun --account=$SBATCH_ACCOUNT --cpus-per-task=2 --mem=32G --time=02:00:00 --pty bash
```

### Compute Nodes

```shell
salloc --time=2:00:00 --cpus-per-task=8 --mem=32GB --account=<pi> --partition=gpu --gres=gpu:1
```

### View Jobs

```
squeue -u $USER
```

### TMUX

Keep a long interactive session alive with tmux:

```shell
tmux new -s nuscenes        # start
# Ctrl-b d                  # detach; the work continues
tmux attach -t nuscenes     # come back
```


---

## Modules

```
module avail
```


### Github

```
module load gcc/13.3.0
```

```
module load gh/2.43.1
```

```
gh auth login
```





