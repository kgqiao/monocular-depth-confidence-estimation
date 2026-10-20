# Open the notebooks on CARC

[`notebooks/cache-tutorial.ipynb`](../notebooks/cache-tutorial.ipynb) is a tour of the cache and the API that reads it. This guide opens it in JupyterLab on CARC, against the cache from [cache-building-on-carc.md](cache-building-on-carc.md). The same steps, without the JupyterLab part, open it on your own computer.

## 1. Register the kernel

JupyterLab runs code in a kernel. The kernel must be the Python of the `confhead` environment, so that `import confhead` works. Register it once, from a CARC login node:

```sh
cd ~/Vision-Project-CSCI-566/confhead
git pull
uv sync --group notebook
uv run python -m ipykernel install --user --name confhead --display-name confhead
```

`uv sync --group notebook` adds JupyterLab and the kernel package to the environment. The last command writes `~/.local/share/jupyter/kernels/confhead/kernel.json`, which points at `.venv/bin/python`. JupyterLab lists it as **confhead**. Run the two commands again after you move or rebuild `.venv`.

To check:

```sh
uv run jupyter kernelspec list
```

## 2. Start JupyterLab

Go to [ondemand.carc.usc.edu](https://ondemand.carc.usc.edu), open **Interactive Apps**, and choose **JupyterLab**. Fill in the form:

| Field | Value |
|---|---|
| Account | your project account, the one in `SBATCH_ACCOUNT` |
| Partition | `main`; the notebook needs no GPU |
| CPUs | 4 |
| Memory | 16 GB. The trainval index takes a few hundred MB; section 11 of the notebook loads the nuScenes tables, which take several GB. |
| Time | 2 hours |

Click **Launch**. When the session starts, click **Connect to Jupyter**.

## 3. Open the notebook

In the file browser on the left, go to `Vision-Project-CSCI-566/confhead/notebooks` and open `cache-tutorial.ipynb`. If JupyterLab asks for a kernel, choose **confhead**. The kernel name is also at the top right of the notebook; click it to change it.

Run the cells from the top. The *Setup* section finds the cache:

- With no change, it opens your own cache, `/scratch1/$USER/confhead-cache/nuscenes-trainval`.
- To open a cache that a teammate built, set `CACHE_ROOT` in the second code cell to their path, for example `"/scratch1/<user>/confhead-cache/nuscenes-trainval"`. The images come from their dataset, which the cache remembers.

## On your own computer

Steps 1 and 3 are the same. Instead of step 2:

```sh
uv run jupyter lab notebooks/cache-tutorial.ipynb
```

With no trainval cache on the machine, the notebook opens the mini cache under `outputs/cache/`, which `examples/build_cache.py` builds from nuScenes v1.0-mini.

## If something goes wrong

| Problem | What to do |
|---|---|
| **confhead** is not in the kernel list | Do step 1 again. Then refresh the JupyterLab page: the list loads when the page opens. |
| `ModuleNotFoundError: No module named 'confhead'` | The notebook runs on another kernel. Choose **confhead** at the top right. |
| `FileNotFoundError: no cache found` | Set `CACHE_ROOT` to the cache path. The path must hold `index.json`; if it does not, the check job of the build has not run yet. |
| `entry.image()` cannot find the image | The dataset is not where the cache was built from. Set `DATAROOT` in the second code cell. |
| The kernel dies in section 11 | The session has too little memory for the trainval tables. Start a JupyterLab session with more, or skip the section. |
