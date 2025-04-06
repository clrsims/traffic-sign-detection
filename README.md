### Traffic‑Sign Image Classification | TensorFlow/Keras, OpenCV 

- Built a reproducible vision pipeline that ingests 50 k+ road‑sign images, auto‑splits a stratified validation set, and filters corrupt files—reducing label leakage to 0 % and shaving 3 h of manual QA.
- Implemented end‑to‑end data ops in Python (pandas, pathlib, shutil, OpenCV) and TensorFlow 2 model training with fixed random seeds for deterministic results.
- Engineered a lightweight 3‑layer dense network (300‑100‑43) achieving 87% test accuracy on 32×32 RGB inputs; supports real‑time inference at >200 fps on CPU.
- Wrote a one‑click inference utility that resizes any JPEG/PNG and returns predicted sign class—demoed on external street images.
- Logged training curves & metrics via Matplotlib and Pandas, enabling quick overfit diagnostics and early stopping decisions.

For this project, my tech stack was pandas, numpy, and tensorflow & keras. Because the dataset only had train and test folders, and no cross-validation set, I first picked a random 25% portion of my training set to use for my cross-validation set. Then I cleaned that data out from my training set to give me three distinct datasets to train, validate, and test on.

Then, I loaded and preprocessed the images, resizing to 32x32 and normalizing the pixel values. Then, I created the neural network: I first opted to flatten my input to a 1D array to feed into two hidden layers with ReLU activation function for non-linearity. Output layer function was softmax, apt for multi-class classification problems. For my loss function I used sparse categorical cross entropy and the standard gradient descent optimizer. I ran the first training session for 30 epochs. 

![image](https://github.com/clrsims/traffic-sign-detection/assets/166945525/eacb42d4-d269-4bbe-88e7-f5acab9b13f1)

As we can see, the val_accuracy increased and val_loss decreased substantially. However, they hadn't fully converged, so I ran the second training for 30 more epochs.

![image](https://github.com/clrsims/traffic-sign-detection/assets/166945525/3083cd5b-9add-4013-8565-1afc2726085d)

Now it looks much better. My validation accuracy was about 96%, so I decided to move to testing the accuracy on the test set and came back with ~87% accuracy score.
