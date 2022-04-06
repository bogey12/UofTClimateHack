# UofTClimateHack

This repository outlines the solutions and code behind the University of Toronto Team's 2nd Place Submission for 2022 [ClimateHack](https://climatehack.ai/).

# Background
Better near-term forecasting of solar electricity generation will enable electricity grid operators around the world to do a better job of scheduling their grids. 

For example, currently the UK National Energy Grid Operator use a combination of solar and natural gas sources to generate power. The objective of the grid is to reliably supply electricity to meet demand at all times, and hence use natural gas on 24/7 standby in the event of sudden falls in solar production (e.g. due to dense cloud coverage). 

By developing better solar forecasting techniques, the Grid Operator could minimize the use of standby gas turbines, potentially leading to a substantial reduction in carbon emissions of up to 100 kilotonnes a year. While it is incredibly difficult to accurately predict the climate impact, a rough estimate suggests that better solar power forecasts if deployed worldwide could reduce global carbon emissions by about 100 million tonnes of CO2 a year by 2030.

# Challenge
We had ~1.5 months as individuals, and 1 week as a team to apply cutting-edge machine learning techniques in order to develop the best satellite imagery prediction algorithm for use in solar photovolatic output forecasting. 

**The specific challenge:** from a series of 12 images covering a 128×128-pixel region cropped out of a series of much larger satellite images taken five minutes apart, accurately predict the next 24 images for the central 64×64-pixel area, corresponding to the next two hours of satellite imagery.

[Open Climate Fix](https://www.openclimatefix.org/) provided us with ~2 years of high resolution satellite imagery over the UK and north-western Europe from EUMETSAT's [Spinning Enhanced Visible and InfraRed Imager Rapid Scanning Service](https://www.eumetsat.int/rapid-scanning-service) with a spatial resolution of about 2-3 km. 

# Our Main Approach
Find a summary of our main approach and other attempted solutions here: [Presentation Slide Deck](https://drive.google.com/file/d/1FbSnPaqEpnLMwjqKs3ynlENwFnrbznGU/view?usp=sharing)

Our main approach uses a 5-level U-Net augmented with a standard vision transformer in the bottleneck to better model global relationships (see UNETViT in submission/model.py). This approach got us 2nd in the leaderboard with a MS_SSIM Score of 0.836. Other improvements from our main approach over a standard U-Net include the use of group normalization instead of batch normalization and depth-wise separable convolutions to reduce model size by 3x without noticably dropping performance.

**Sample predictions from our model:**
![](https://github.com/bogey12/UofTClimateHack/blob/TonyDev/anim_UNet_VIT/2.gif)
Shown above is a GIF of the model's predictions compared with the ground truth. The top row compares the raw image sequence, whereas the bottom row compares their Fast Fourier Transforms (FFTs). The last column shows the difference between the two images. 

FFTs decompose signals into a linear combination of pure frequency components. This was applied to the predictions to help determine quantitatively which types of images the model was failing to predict. In the FFT diagram, low frequency features are shown in the center, with the frequency of the features increasing as you move away from it.

This analysis shows that the model predicts low frequency features well but fails to predict mid to high frequency features, as shown by the fact that the model's FFT is significantly more concentrated in the center compared to the ground truth.

# Other Solutions and Ideas
**Trajectory GRU:** Trajectory GRU is a model architecture that improves on previous attempts at incorporating convolutions in autoregressive models (ConvLSTM, ConvGRU). These past attempts use fixed convolutions on the hidden states at each time step, which means that connections between the current hidden state and past hidden state are always the same. This is not desirable when predicting fast moving objects, as the past hidden state representations of these objects should ideally have their information propagated to a different location in the future hidden state representation. TrajGRU solves this problem by dynamically generating these connections.

Code Reference: [TrajGRU](https://github.com/Hzzone/Precipitation-Nowcasting)

**Convolution Based Attention Module (CBAM):** First, max and average pooling to aggregate spatial information and then attention is computed across input channels to capture inter-channel dependencies. This module was included in some of our U-Net-based experiments in our convolution blocks. These modules weren't included in our final model because they provided minimal benefit to performance.

Code Reference: [CBAM](https://github.com/elbuco1/CBAM)

**Focal Transformers:** Focal transformer is a multi-resolution vision transformer architecture suited for dealing with high-resolution images. This architecture attends at a fine-resolution locally and course resolution globally allowing focal transformers to capture high-resolution local information while maintaining scalability to larger input features. 

We used focal transformers as both the encoder and decoder replacement in our UNET architectures similar to how a Swin-UNet is organized. Our experiments were promising, but we ran out of time to train our focal transformer experiments to convergence.

Code Reference: [Focal Transformer](https://github.com/microsoft/Focal-Transformer)
