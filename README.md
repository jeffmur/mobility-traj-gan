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
| 7/17 | Added LSTMTrajGAN and MARC models       | alexkyllo |

## TODOs

- [x] Port LSTM-TrajGAN to TensorFlow 2 so it can be run in the same environment
- [x] Train LSTM-TrajGAN on Foursquare NYC dataset
- [x] Preprocessing code for MDC data so it can be fed into LSTM-TrajGAN
- [x] Train LSTM-TrajGAN on MDC dataset
- [x] Port MARC reidentifier model to TF2
- [x] Train MARC on MDC dataset
- [ ] Compare MARC performance on real vs. generated MDC trajectories for LSTM-TrajGAN
- [x] Preprocessing code for GeoLife data so it can be fed into LSTM-TrajGAN
- [ ] LSTM-TrajGAN training on GeoLife dataset
- [ ] Preprocessing code for Privamov data so it can be fed into LSTM-TrajGAN
- [ ] LSTM-TrajGAN training on Privamov dataset
- [x] Post-processing code to output LSTM-TrajGAN generated trajectories to CSV
- [ ] Get outputs from Yuting's LSTM-AE model on MDC, FourSquare, Privamov and GeoLife datasets
- [x] Train MARC on FourSquare NYC dataset
- [ ] Train MARC on GeoLife dataset
- [ ] Train MARC on Privamov dataset
- [ ] Compare MARC performance on real vs. generated trajectories for LSTM-AE
