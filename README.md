# SceneScript Demo
This repository contains inference code for [SceneScript](https://www.projectaria.com/scenescript) with visualisations.


## Installation

The code can be installed via `conda`. Please follow the instructions [here](https://docs.anaconda.com/anaconda/install/index.html) to install Anaconda for your machine.

We list our dependencies in `environment.yaml` file. To install the dependencies and create the env, run:

```
conda env create --file=environment.yaml

conda activate scenescript
```

## Models
We provide two trained models that operate on semi-dense point clouds obtained via [Project Aria Machine Perception Services](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_pointcloud):

* The first one is trained on [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/) (ASE). This model predicts layout (wall/door/window) and gravity-aligned 3D bounding boxes with class labels. Note that as of 9/17/2024, ASE has not been publicly released with bounding box annotations.
* The second model is trained on a proprietary dataset that is quite similar to ASE. Similar to the first model, it predicts both layout (wall/door/window) and gravity-aligned 3D bounding boxes. However, because this dataset is proprietary, we do not release the bounding box class taxonomy, and this model does not predict classes for the bounding boxes. This proprietary dataset contains non-Manhattan layout configurations and thus generalises better to real-world scenes, and was used in the demo videos as seen on the [SceneScript website](https://www.projectaria.com/scenescript/).

| Training Dataset | Download link |
| -------- | -------- |
| Aria Synthetic Environments | [scenescript_website](https://www.projectaria.com/scenescript/) |
| Internal Proprietary Dataset | [scenescript_website](https://www.projectaria.com/scenescript/) |

## Example Data

The provided SceneScript models operate on semi-dense point clouds obtained via [Project Aria Machine Perception Services](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_pointcloud). To obtain a few examples from [ASE](https://www.projectaria.com/datasets/ase/) (synthetic) and [Aria Everyday Activities](https://www.projectaria.com/datasets/aea/) (real-world), run the following commands:

```
export SEMIDENSE_SAMPLE_PATH=/tmp/semidense_samples
mkdir -p $SEMIDENSE_SAMPLE_PATH/ase
mkdir -p $SEMIDENSE_SAMPLE_PATH/aea

export ASE_BASE_URL="https://www.projectaria.com/async/sample/download/?bucket=ase&filename="
export AEA_BASE_URL="https://www.projectaria.com/async/sample/download/?bucket=aea&filename="
export OPTIONS="-C - -O -L"

# ASE (Synthetic)
curl -o $SEMIDENSE_SAMPLE_PATH/ase/ase_examples.zip $OPTIONS "${ASE_BASE_URL}ase_examples.zip"

# AEA (Real-world)
curl -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1.zip $OPTIONS "${AEA_BASE_URL}loc1_script1_seq1_rec1.zip"
curl -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample.zip $OPTIONS "${AEA_BASE_URL}loc1_script2_seq1_rec1_10s_sample.zip"
curl -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample.zip $OPTIONS "${AEA_BASE_URL}loc1_script2_seq1_rec2_10s_sample.zip"

# Unzip everything
unzip -o $SEMIDENSE_SAMPLE_PATH/ase/ase_examples.zip -d $SEMIDENSE_SAMPLE_PATH/ase/ase_examples
unzip -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1.zip -d $SEMIDENSE_SAMPLE_PATH/aea/loc1_script1_seq1_rec1
unzip -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample.zip -d $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec1_10s_sample
unzip -o $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample.zip -d $SEMIDENSE_SAMPLE_PATH/aea/loc1_script2_seq1_rec2_10s_sample
```

## Jupyter Notebook

See the [inference Jupyter Notebook](inference.ipynb) for an example of how to run the network. In order to run this, Jupyter must be installed (this is included in `environment.yaml`). If you haven't used Jupyter Notebooks before, [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) is a tutorial to get you up to speed.

Notes:

* Make sure to activate the conda environment before running jupyter. This can be done with ```conda activate scenescript; jupyter notebook```
* Make sure to update the filepaths to point to the downloaded files.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing SceneScript

If you use SceneScript in your research, please use the following BibTeX entry.

```
@inproceedings{avetisyan2024scenescript,
    title       = {SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model},
    author      = {Avetisyan, Armen and Xie, Christopher and Howard-Jenkins, Henry and Yang, Tsun-Yi and Aroudj, Samir and Patra, Suvam and Zhang, Fuyang and Frost, Duncan and Holland, Luke and Orme, Campbell and Engel, Jakob and Miller, Edward and Newcombe, Richard and Balntas, Vasileios},
    booktitle   = {European Conference on Computer Vision (ECCV)},
    year        = {2024},
}
```
