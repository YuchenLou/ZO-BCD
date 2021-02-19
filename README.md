# ZO-BCD
Codes from the paper "A Zeroth-Order Block Coordinate Descent Algorithm for Huge-Scale Black-Box Optimization"

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](danielmckenzie.github.io)


and put the `imgs` folder in `../Adversarial_Attacks_Experiments`. This path can be changed
in `ZO_BCD_Wavelet_Attack_Test.m`.

To prepare the Audio Commands dataset, download and unzip the following archive:

[Audio Test Set](danielmckenzie.github.io)


and put the folders in `../Sounds`. This path can be changed
in `ZO_BCD_Audio_Targeted_Test.m`.


Attack on imagenet for 1000 images (in the folder Image Attack):

run 
```
ZO_BCD_Wavelet_Attack_Test.m


Attack on audio commands (in the folder Audio Attack):

run 
```
ZO_BCD_Audio_Targeted_Test.m
