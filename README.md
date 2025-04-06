### Traffic‑Sign Image Classification | TensorFlow/Keras, OpenCV 

- Built a reproducible vision pipeline that ingests 50 k+ road‑sign images, auto‑splits a stratified validation set, and filters corrupt files—reducing label leakage to 0 % and shaving 3 h of manual QA.
- Implemented end‑to‑end data ops in Python (pandas, pathlib, shutil, OpenCV) and TensorFlow 2 model training with fixed random seeds for deterministic results.
- Engineered a lightweight 3‑layer dense network (300‑100‑43) achieving 87% test accuracy on 32×32 RGB inputs; supports real‑time inference at >200 fps on CPU.
- Wrote a one‑click inference utility that resizes any JPEG/PNG and returns predicted sign class—demoed on external street images.
- Logged training curves & metrics via Matplotlib and Pandas, enabling quick overfit diagnostics and early stopping decisions.

## Dataset

The data used is from the **[German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)** dataset on Kaggle. It includes:

- 'train.csv' - image paths and labels for training
- 'test.csv' - image paths and labels for evaluation
- 'meta.csv' - class metadata (e.g. sign names, image references)
- 'Train/' - image folders: 43 subfolders (0 to 42), each containing labeled .png images
- 'internet_pictures/' - folder with external images for prediction demo

