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

## Share your scratch folder

Only you can open your scratch folder. The course group `yzhao010_1531` has about 170 members, so group permissions would share the folder with the whole course. Use ACLs instead: they give access to named users only. The commands below name all five team members (`naday`, `tawde`, `kgqiao`, `ncbao`, `ajarteag`), so you can paste them as they are. The entry that names you has no effect.

1. Check that the folder is private:

   ```sh
   ls -ld /scratch1/$USER
   ```

   The mode must be `drwx------`. If it is not, run `chmod go-rwx /scratch1/$USER`.

2. Let the team pass through the folder. With `x` and no `r`, they can open paths inside it but cannot list it:

   ```sh
   setfacl -m u:naday:x,u:tawde:x,u:kgqiao:x,u:ncbao:x,u:ajarteag:x /scratch1/$USER
   ```

3. Give the team read access to each folder you share. The command visits every file, so run it as a job:

   ```sh
   sbatch --partition=main --cpus-per-task=1 --mem=2G --time=12:00:00 \
     --job-name=share-scratch --output=$HOME/share-scratch-%j.out \
     --wrap="setfacl -R -m u:naday:rX,u:tawde:rX,u:kgqiao:rX,u:ncbao:rX,u:ajarteag:rX /scratch1/$USER/nuscenes"
   ```

   `X` gives execute permission to folders only. Files you add later do not get these permissions, so run the job again after you add data.

4. Ask a teammate to check:

   ```sh
   namei -l /scratch1/<usc-id>/nuscenes
   ```

   It prints the permissions of each folder on the path. Access stops at the first folder that lacks `x` for them.

After step 2, `ls -ld` shows a mode like `drwx--x---+`. The `+` marks an ACL, and the group position now shows the ACL mask, an upper limit on the named-user entries. The course group still has no access: `getfacl /scratch1/$USER` shows `group::---`. Avoid `chmod g-...` on shared folders, because it lowers the mask and removes the team's access.

## Get the code

On a login node:

```sh
git clone https://github.com/kgqiao/monocular-depth-confidence-estimation.git ~/Vision-Project-CSCI-566
```

## Next steps

- [Build the cache on CARC](../confhead/guides/cache-building-on-carc.md)
- [Open the notebook on CARC](../confhead/guides/notebook-on-carc.md)
- [Back up the cache on Hugging Face](../confhead/guides/cache-backup-on-huggingface.md)
- [CARC user guides](https://www.carc.usc.edu/user-guides)
