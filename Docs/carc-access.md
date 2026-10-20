# Access USC CARC

The team runs its large jobs on Discovery, the USC CARC cluster. Jobs include the nuScenes download, the cache build, and the Hugging Face upload. This page tells you how to connect.

## Before you connect

Connect your computer to the USC on-campus Wi-Fi, or to the USC VPN with Cisco AnyConnect. Discovery does not accept connections from other networks.

## Log in with SSH

```sh
ssh <usc-id>@10.72.0.13
```

Type your USC password when the prompt asks for it. You are now on a login node.

To type `ssh carc` instead, add these lines to `~/.ssh/config` on your computer:

```
Host carc
    HostName 10.72.0.13
    User <usc-id>
```

Use the same name to copy files to and from the cluster:

```sh
scp results.csv carc:~/
scp carc:~/results.csv .
```

> [!IMPORTANT]
> Login nodes are for short commands: `git pull`, `uv sync`, editing files, and submitting jobs. Run long or heavy work as a Slurm job with `sbatch`.

## Use the web dashboard

[CARC OnDemand](https://ondemand.carc.usc.edu) gives you a file browser, a shell, and interactive apps such as JupyterLab in your web browser. The same network rule applies.

## Folders

| Folder | What it holds |
|---|---|
| `~/` | Your home folder. Clone the repository here. |
| `/project2/yzhao010_1531` | The course project folder, shared with the course. |
| `/scratch1/<usc-id>` | Your scratch folder: the fastest storage, for large data. The nuScenes dataset and the cache are in `/scratch1/naday`. |

## Get the code

On a login node:

```sh
git clone https://github.com/kgqiao/Vision-Project-CSCI-566.git ~/Vision-Project-CSCI-566
```

## Next steps

- [Build the cache on CARC](../confhead/guides/cache-building-on-carc.md)
- [Open the notebook on CARC](../confhead/guides/notebook-on-carc.md)
- [Back up the cache on Hugging Face](../confhead/guides/cache-backup-on-huggingface.md)
- [CARC user guides](https://www.carc.usc.edu/user-guides)
