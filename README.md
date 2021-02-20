# ZO-BCD
Codes from the paper "A Zeroth-Order Block Coordinate Descent Algorithm for Huge-Scale Black-Box Optimization"

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](danielmckenzie.github.io)


and put the `imgs` folder in `../Adversarial_Attacks_Experiments`. This path can be changed
in `ZO_BCD_Wavelet_Attack_Test.m`.

## Audio Attack
To replicate our audio attacks do the following:
    1. You will need the Audio and Deep Learning toolboxes in Matlab installed.
    2. Follow the instructions [here](https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html) to download the SpeechCommands data set. Put the folders in `../Sounds`. This path can be changed
    in `ZO_BCD_Audio_Targeted_Test.m`.
    3. Open the MATLAB Deep Learning Speech Recognition Example folder (as described [here](https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html) ). You will need to copy commandNet.mat and helperExtractAuditoryFeatures.m into the Audio Attack folder.
    4. That's it! You can now run ZO_BCD_Audio_Targeted_Test.m and see the results.

To prepare the Audio Commands dataset, download and unzip the following archive:

[Audio Test Set](danielmckenzie.github.io)


and put t


Attack on imagenet for 1000 images (in the folder Image Attack):

run 
```
ZO_BCD_Wavelet_Attack_Test.m
```

Attack on audio commands (in the folder Audio Attack):

run 
```
ZO_BCD_Audio_Targeted_Test.m
```
