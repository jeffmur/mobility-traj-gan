# Traj-GAN applied to CDR

## Purpose

Processing mobile GPS location time series data with an LSTM Autoencoder network
to discover user communities through trajectory clustering, in a federated
environment with privacy preservation.

## How-to...

### Install the dependencies

This is a TensorFlow 2 project and if you plan to train models on an
NVIDIA GPU we recommend using conda to install the dependencies
because each version of TensorFlow only works with a specific version
of CUDA (see [TensorFlow docs for tested
configurations](https://www.tensorflow.org/install/source#linux)) and
conda can install isolated CUDA versions in its environments to
prevent conflicts.

```
conda env create -f environment.yml
conda activate
```

### Add a dataset

1. Add two environment variables to a `.env` file in this (root)
   directory: `xxx_INPUT_DIR` which is the absolute directory path of
   the raw input dataset on your system, and `xxx_INPUT_FILE`, which
   is a file location where you want preprocessed data to be saved,
   i.e. a CSV file. Replace the `xxx` with a 3-letter "nickname" for
   your dataset.
2. Create a new .py file in [src/datasets/](src/datasets/) for your
   dataset.
3. Write a class that subclasses `Dataset` (from
   [src/datasets/base.py](src/datasets/base.py)) and implements a
   `preprocess()` method that reads in the raw data and returns a
   `pandas.DataFrame`.
4. Import your class into
   [src/datasets/__init__.py](src/datasets/__init__.py) and add the
   class name to the `DATASETS` list.

### Add a model

1. Create a new .py file in [src/models/](src/models/) and write a
   class that inherits from `TrajectoryModel` (in
   [src/models/base.py](src/models/base.py)) and implements at least
   `train`, `predict`, `save` and `restore`
   abstract methods. If it's a supervised model (like MARC) then
   you'll also want to add an `evaluate` method to get metrics on the
   test set.
2. Import your model class into
   [src/models/__init__.py](src/models/__init__.py) and add the class
   name to the `MODELS` list.

### Train a model on a dataset

Use the CLI script's `train` command:

```
$ python mobility_cli.py
Usage: mobility_cli.py [OPTIONS] COMMAND [ARGS]...

  Command line interface for the mobility learning framework.

Options:
  --help  Show this message and exit.

Commands:
  evaluate  Use SAVED_MODEL to predict the labels of DATASET.
  predict   Use trained MODEL saved in SAVED_PATH to make predictions based...
  train     Train MODEL on DATASET stored in DATASET_PATH for EPOCHS.

$ python mobility_cli.py train --help
Usage: mobility_cli.py train [OPTIONS] [LSTMTrajGAN|MARC] [MDCLausanne|GeoLife
                             Beijing|FourSquareNYC|PrivamovLyon] EPOCHS

  Train MODEL on DATASET stored in DATASET_PATH for EPOCHS.

Options:
  --help  Show this message and exit.

$ python mobility_cli.py train LSTMTrajGAN GeoLifeBeijing 200
```

### Generate predictions from a model

Use the CLI script's `predict` command:

```
$ python mobility_cli.py predict --help
Usage: mobility_cli.py predict [OPTIONS] [LSTMTrajGAN|MARC] SAVED_PATH [MDCLau
                               sanne|GeoLifeBeijing|FourSquareNYC|PrivamovLyon
                               ] OUTPUT_PATH

  Use trained MODEL saved in SAVED_PATH to make predictions based on DATASET
  and write to OUTPUT_PATH as CSV.

Options:
  --help  Show this message and exit.

$ python mobility_cli.py predict LSTMTrajGAN LSTMTrajGAN experiments/LSTMTrajGAN_GeoLifeBeijing/2021-07-24T23:55:37/saved_model/ outputs/LSTMTrajGAN_GeoLifeBeijing_predictions.csv
```

### Evaluate a model

Use the CLI script's `evaluate` command:

```
$ python mobility_cli.py evaluate --help
Usage: mobility_cli.py evaluate [OPTIONS] [LSTMTrajGAN|MARC] SAVED_PATH [MDCLa
                                usanne|GeoLifeBeijing|FourSquareNYC|PrivamovLy
                                on]

  Use SAVED_MODEL to predict the labels of DATASET.

Options:
  --help  Show this message and exit.
```

## Change Log

---

| Date | Note                                    | Author    |
| ---- | --------------------------------------- | --------- |
| 3/31 | Created Repo; initalization & config.py | jeffmur   |
| 4/1  | Project structure and importing fixes   | alexkyllo |
| 4/5  | Optimize freqMatrix function            | alexkyllo |
| 4/6  | Appended Ali's LSTM-AE, updated req.txt | jeffmur   |
| 7/17 | Added LSTMTrajGAN and MARC models       | alexkyllo |
| 7/24 | GeoLife and Privamov Ready for training | jeffmur   |
| 7/25 | Add training instructions to README     | alexkyllo |

## TODOs

- [x] Port LSTM-TrajGAN to TensorFlow 2 so it can be run in the same environment
- [x] Preprocessing code for MDC data so it can be fed into LSTM-TrajGAN
- [x] Port MARC reidentifier model to TF2
- [x] Preprocessing code for GeoLife data so it can be fed into LSTM-TrajGAN
- [x] Preprocessing code for Privamov data so it can be fed into LSTM-TrajGAN
- [x] Post-processing code to output LSTM-TrajGAN generated trajectories to CSV
- [x] Train LSTM-TrajGAN on MDC Lausanne dataset
- [x] Train LSTM-TrajGAN on Foursquare NYC dataset
- [x] Train LSTM-TrajGAN on GeoLife Beijing dataset
- [x] Train LSTM-TrajGAN on Privamov Lyon dataset
- [x] Train MARC on MDC dataset
- [ ] Train MARC on FourSquare NYC dataset
- [ ] Train MARC on GeoLife dataset
- [ ] Train MARC on Privamov dataset
- [ ] Compare MARC performance on real vs. generated MDC trajectories for LSTM-TrajGAN
- [ ] Get outputs from Yuting's LSTM-AE model on MDC, FourSquare, Privamov and GeoLife datasets
- [ ] Compare MARC performance on real vs. generated trajectories for LSTM-AE
