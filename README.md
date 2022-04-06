# UofTClimateHack

This repository outlines the solutions and code behind the University of Toronto Team's 2nd Place Submission for 2022 [ClimateHack](https://climatehack.ai/).

# Background
Currently, UK energy grids use a combination of both solar and natural gas sources to generate power. Due to a lack of accurate solar forecasting, natural gas turbines are on all the time, with solar power stored as excess in reserves. If they had more accurate solar forecasting algorithms, they could turn on/off natural gas sources when solar production is adequate. 

This would allow them to minimize the use of standby gas turbines, potentially leading to a substantial reduction in carbon emissions of up to 100 kilotonnes a year. While it is incredibly difficult to accurately predict the climate impact, a rough estimate suggests that better solar power forecasts if deployed worldwide could reduce global carbon emissions by about 100 million tonnes of CO2 a year by 2030.

# Challenge
We had ~1.5 months as individuals, and 1 week as a team to apply cutting-edge machine learning techniques in order to develop the best satellite imagery prediction algorithm for use in solar photovolatic output forecasting. **The specific challenge:** from a series of 12 images covering a 128×128-pixel region cropped out of a series of much larger satellite images taken five minutes apart, accurately predict the next 24 images for the central 64×64-pixel area, corresponding to the next two hours of satellite imagery.

# Our Main Approach
Find a summary of our main approach and other attempted solutions here: (INSERT SLIDES LINK HERE)

Our main approach uses a 5-level U-Net augmented with a standard vision transformer in the bottleneck to better model global relationships (see UNETViT in submission/model.py). This approach got us 2nd in the leaderboard with a MS_SSIM Score of 0.836. Other improvements from our main approach over a standard U-Net include the use of group normalization instead of batch normalization and depth-wise separable convolutions to reduce model size by 3x without noticably dropping performance.

Sample predictions from our model:
![](https://github.com/bogey12/UofTClimateHack/blob/TonyDev/anim_UNet_VIT/0.gif)
![](https://github.com/bogey12/UofTClimateHack/blob/TonyDev/anim_UNet_VIT/2.gif)
![](https://github.com/bogey12/UofTClimateHack/blob/TonyDev/anim_UNet_VIT/4.gif)
![](https://github.com/bogey12/UofTClimateHack/blob/TonyDev/anim_UNet_VIT/6.gif)
(EXPLAIN A LITTLE BIT ABOUT THE PREDICTIONS + FTT stuff)

# Other Solutions and Ideas
**Trajectory GRU:** (ANDY FILL OUT TRAJ GRU SECTION HERE)

**Convolution Based Attention Module (CBAM):** First, max and average pooling to aggregate spatial information and then attention is computed across input channels to capture inter-channel dependencies. This module was included in some of our U-Net-based experiments in our convolution blocks. These modules weren't included in our final model because they provided minimal benefit to performance.

Code Reference: https://github.com/elbuco1/CBAM

**Focal Transformers:** Focal transformer is a multi-resolution vision transformer architecture suited for dealing with high-resolution images. This architecture attends at a fine-resolution locally and course resolution globally allowing focal transformers to capture high-resolution local information while maintaining scalability to larger input features. 

We used focal transformers as both the encoder and decoder replacement in our UNET architectures similar to how a Swin-UNet is organized. Our experiments were promising, but we ran out of time to train our focal transformer experiments to convergence.

Code Reference: https://github.com/microsoft/Focal-Transformer