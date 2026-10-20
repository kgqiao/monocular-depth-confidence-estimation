# Knowing When Not to Trust Depth

The CSCI 566 group project, Fall 2026. Depth Anything V2 gives a depth for every pixel of a camera image, but it does not say which pixels to trust. We train a small per-pixel confidence head on top of the frozen model, with lidar labels and radar inputs from nuScenes.

## Quick Links

- [Access USC CARC](Docs/carc-access.md): connect to the cluster with SSH or OnDemand, and find the team folders.
- [Cache on Hugging Face](https://huggingface.co/datasets/nadayusc/confhead-cache): a copy of the confidence head input cache, for the team to download.
- [nuScenes](https://www.nuscenes.org/nuscenes): the driving dataset that the project uses.
- [Cache tutorial notebook](confhead/notebooks/cache-tutorial.ipynb): a tour of the cache and the API that reads it. To open it on CARC, see [notebook-on-carc.md](confhead/guides/notebook-on-carc.md).
- [Project wiki](Docs/Wiki/CS566-project.md): an overview of the project, the dataset plan, and the training design.
- [Training team work log](Docs/worklog.md): what the training team has done, by date.

## Contents

- [`confhead/`](confhead/README.md): the Python library for the project: dataset loading, the frozen model, the cache of its outputs, labels, and the training dataset.
- [`Docs/Prepreposal/`](Docs/Prepreposal/): the brainstorms, the pre-proposal, and its source.
- [`Docs/proposal-vs-pipeline.md`](Docs/proposal-vs-pipeline.md): what the proposal asks for against what `confhead` does.
- [`Docs/Papers/`](Docs/Papers/): background papers.
- [`fair-matching/`](fair-matching/README.md): the script that assigned the team roles.
- `datasets/`: the datasets, on each machine. Not in git. `confhead` expects nuScenes v1.0-mini at `datasets/nuscenes/v1.0-mini`.
