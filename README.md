# ColorBlindZebra
Introduction

We are trying to colorize black and white images. We are interested in what images that are black and white would look like in color, especially old images which weren't originally captured with color.

The paper's purpose is to train a conditional GAN for image colorization and transfer that knowledge to multilabel image classification and semantic segmentation. We are using this paper because it gives a good model of what architecture to follow.

It's worth noting that we are only going to implement the GAN, and ignore the rest of the paper. This is a self-supervised regression problem since we are trying to predict colored pixels and minimize the MSE.

Related Work

We looked at another application (https://cs230.stanford.edu/projects_fall_2019/reports/26259829.pdf) of GANs used for structured prediction, this time, predicting the stock market price of companies. The paper looked at using GAN, LSTM, and traditional time series models. They concluded that GANs can make very accurate predictions of time series data, and there were no significant advantages with using a GAN over models traditionally used like LSTM or other time series analysis.

Data


We plan on using ImageNet for our data. The data is 14 million images of size 256x256. We will most likely have to cut down the number of images we use, but are not entirely sure how small our data set will get. We may also explore scraping images from Instagram, Facebook, etc to use for our data.

Methodology


Our architecture will be a conditional GAN. We are training our model using Image Net images on a mix of google cloud and the department machines. We think the hardest part of the paper will be optimization to make our model perform well.

Metrics


We will measure success with how plausible the image colorization is. We plan to look at old images which are in black and white to see what they might've looked like with color, and we may scrap images from instagram and see what those look like in color. We plan on using PSNR to (peak signal to noise ratio) as our metric to measure the quality of the image reconstruction. The paper also mentions another method SSIM (structural similarity index measure) to take into account factors such as luminance, contrast, and structural change between two images. They were able to reach an average PSNR score of 20.94, and an average SSIM score of .85

Our base goal is to be able to colorize black and white images in a semi plausible way. Our target goal is to be able to colorize images where human eyes are not able to tell if they were captured in color, or if the image was black and white and has been colorized. Our stretch goal is to get a low PSNR, where our model is almost able to perfectly colorize images. Another stretch goal would be getting an SSIM average of .9 or above.

Ethics


What broader societal issues are relevant to your chosen problem?

An area our problem could be relevant to is coloring security camera footage. Often it is in black and white, and if we can accurately color the image it could help whoever needs footage. One thing we need to be careful about is the general bias against minorities in regards to computer vision. For example, if our recoloring process has a hard time deciding what color to paint an individual, then it could perhaps create false evidence, and contain biases against minorities that could negatively harm people, either in the criminal justice system or elsewhere. Thus, we need to be careful about building in biases to our algorithm.

Additionally, if the photograph/footage is recolored incorrectly, this could cause discrepancies within any future software, algorithm, etc it is fed into. For example, if this algorithm was used in museums to recolor old photographs, distorted or incorrectly colored objects/people could lead to historically inaccurate depictions.

Furthermore, there is great opposition to colorization as a whole amongst artists, historians, etc. Some believe the act reverses “longstanding assumptions about the art practices that are closest to reality” (https://petapixel.com/2021/05/18/the-controversial-history-of-colorizing-black-and-white-photos)

How are you planning to quantify or measure error or success? What implications does your quantification have?

We plan on quantifying success by comparing the number of pixels correctly colored out of the total number of pixels in the picture. This does have the implication that perhaps we wrongly color the most important part of the image, but get the background of the image correct, making us think we did this picture very well, when in fact we may not have.

This can lead to the problem of misrepresenting what the actual picture was, and thus we need to be careful and perhaps even change our method of quantifying success to avoid this problem. We could perhaps have some sort of way of figuring out the most critical pixels and weighing those more than the others.

Division of Labor


We will most likely work together on all parts of the project. No strict division of labor has been formed yet.
