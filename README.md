# ZO-BCD
This repository is for our paper:

[1] HanQin Cai, Yuchen Lou, Daniel Mckenzie, and Wotao Yin. A Zeroth-Order Block Coordinate Descent Algorithm for Huge-Scale Black-Box Optimization. In *International Conference on Machine Learning*, 2021.

## Imagenet Attack
To replicate our imagenet attacks do the following:
1. You will need the Wavelet toolbox and the Deep Learning toolbox installed on Matlab.
2. Download the pre-trained Inceptionv3 model, as described [here](https://www.mathworks.com/help/deeplearning/ref/inceptionv3.html).
3. Download the ImageNet test set, available [here](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/). Copy this folder into `Image Attack`. (Alternatively, update the path in `ZOBCD_Image_Attack.m` file).
4. That's it! Run `ZOBCD_Image_Attack.m` to perform the attacks. Note that we have included a few sample outputs in `Examples`.

## Audio Attack
To replicate our audio attacks do the following:
1. You will need the Audio, Wavelet, and Deep Learning toolboxes in Matlab installed (need the version to be at least 2019b for Audio toolbox).
2. Follow the instructions [here](https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html) to download the SpeechCommands data set. Put the folders in `Audio Attack/Sounds`. This path can be changed in `ZO_BCD_Audio_Targeted_Test.m`.
3. Open the MATLAB Deep Learning Speech Recognition Example folder (as described [here](https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html) ). You will need to copy `commandNet.mat` and `helperExtractAuditoryFeatures.m` into `Audio Attack` .
4. That's it! You can now run `ZO_BCD_Audio_Targeted_Test.m` and see the results. Note that we have included a few sample outputs in `Examples`.

## Generic ZO-BCD
We provide both MATLAB and Python Implementations of the generic ZO-BCD in the folders `matlabZO-BCD` and `pyZO-BCD` respectively. We use the MATLAB version for the sythetic experiments in our paper. Run `Test_ZOBCD.m` in the folder `matlabZO-BCD` to get a general comparsion between two variants of ZO-BCD: ZO-BCD-R and ZO-BCD-RC.

Notice that we use MATLAB's "run and time" for the time experiment in our paper.
