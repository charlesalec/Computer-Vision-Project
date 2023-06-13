# Adding a trainable pre-processing to U-Net

## Goals
Our goal was to make a trainable pre processing pipeline for U-Net.

## Motivations
- Why we did what we did?
- What are we trying to achieve?

## Architecture
For this implementation we attempted to connect two different architectures together. The first is a trainable pre-processing pipeline that was used in the past by [1] order to attempt the make object detectors see better in the dark. The second architecture is a vanilla [2] U-Net that we trained in semantic segmentation with the Cityscapes dataset [3].

These 2 architectures are connected in the following way:
```
output1 = PreNet(imagergb)
output2 = F.pad(input=output1, pad=(2, 2, 2, 2))
output = UNet(output2)
```
- Explain how the 2 architectures connect

### Pre-Net [1]
<figure><img src="images/image-1.png" alt="Trulli" style="width:100%"><figcaption align = "center"><b>Pre-processing pipeline overview</b></figcaption></figure>
<figure><img src="images/image.png" alt="Trulli" style="width:100%"><figcaption align = "center"><b>Diagram of the block in the pre-processing</b></figcaption></figure>
For our pre-processing pipeline we have taken inspiration from the Proposed pre-processing pipeline from [1]. It consists of several layers of convolutions, Leaky ReLU together with Max pooling, and ending in a Multi Layer Perceptron (MLP) layer. This MLP layer is the only thing that makes the ConvWB and ConvCC blocks different, as they have a different number of outputs (3 and 9 respectively).These outputs are then applied to the pixel values of the image in the following way
<figure><img src="images/image-2.png" alt="Trulli" style="width:100%"><figcaption align = "center"><b>How the output of the ConvWB block is applied to the colors of the image</b></figcaption></figure>

<figure><img src="images/image-3.png" alt="Trulli" style="width:100%"><figcaption align = "center"><b>How the output of the ConvCC block is applied to the colors of the image</b></figcaption></figure>

Finally the new image is fed into the Shallow ConcNet block and the output is then a new image that should be easier for U-Net to preform image segmentation on.

- Explain our Pre-Net
- Reference GenISP

### U-Net
 - Explain our U-Net

## Training Procedure
- How did we train it?
- What weights did we use?

## Results
- What results did we get?
- Why do we think it acts like that?
- How can they be interpreted?

## Next Step
- Future work


# References:
[1] Morawski, I., Chen, Y. A., Lin, Y. S., Dangi, S., He, K., & Hsu, W. H. (2022). Genisp: Neural isp for low-light machine cognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 630-639).

[2] !TODO: Add U-Net reference

[3] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., ... & Schiele, B. (2016). The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3213-3223).