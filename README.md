# mobility-lstm

## Purpose

Processing mobile GPS location time series data with an LSTM Autoencoder network
to discover user communities through trajectory clustering, in a federated
environment with privacy preservation.

## Change Log

---

| Date | Note                                    | Author    |
| ---- | --------------------------------------- | --------- |
| 3/31 | Created Repo; initalization & config.py | jeffmur   |
| 4/1  | Project structure and importing fixes   | alexkyllo |
| 4/5  | Optimize freqMatrix function            | alexkyllo |
| 4/6  | Appended Ali's LSTM-AE, updated req.txt | jeffmur   |

## TODOs


- [x] Port LSTM-TrajGAN to TensorFlow 2 so it can be run in the same environment
- [x] Train LSTM-TrajGAN on Foursquare NYC dataset
- [x] Preprocessing code for MDC data so it can be fed into LSTM-TrajGAN
- [ ] Train LSTM-TrajGAN on MDC dataset
- [ ] Preprocessing code for GeoLife data so it can be fed into LSTM-TrajGAN
- [ ] LSTM-TrajGAN training on GeoLife dataset
- [ ] Post-processing code to output LSTM-TrajGAN generated trajectories to CSV
- [ ] Implement Yuting's LSTM-AE model
- [ ] Train LSTM-AE model on MDC dataset
- [ ] Train LSTM-AE model on FourSquare NYC dataset
- [ ] Train LSTM-AE model on GeoLife dataset
- [ ] Implement MARC reidentifier model (in TF2)
- [ ] Train MARC on MDC dataset
- [ ] Train MARC on FourSquare NYC dataset
- [ ] Train MARC on GeoLife dataset
- [ ] Compare MARC performance on real vs. generated trajectories for LSTM-TrajGAN
- [ ] Compare MARC performance on real vs. generated trajectories for LSTM-AE
