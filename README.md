# Quantum-internship

This test task consisted of four tasks. The first two of them were for knowledge of Python basics.<br><br>
In the 3rd task *Regression on the tabular data*, you had to build a regression model for anonymous tabular data.<br>
During its completion, I analyzed the data and found that only three features had sufficient significance to build a regression. And only one of them showed a clear nonlinear correlation with the target data. <br>
The regression was performed using a linear regressor with l2 regularization.<br>
The data were passed through the pipeline that generated Polynomial Features and scaled using StandardScaler.<br>
To speed up the training, ExtraTreesRegressor was used to identify significant features.<br>

The fourth task was to detect signs of erosion in soil images.<br>
The data were the image itself and a set of masks. To perform the segmentation, I overlaid the coordinates of the soil erosion polynomials detected in the mask on the image. Then I created a small convolutional network for binary image segmentation. To train the network, I sliced the raster image and the mask into a large number of small images, scaled them in the range of 0-1, and passed them through the network.<br>
Unfortunately, my network did not produce the expected results. This may be partly due to the fact that I was not able to overlay the mask on the bitmap well, as a result, only 435 out of 936 masks were successfully overlaid and as a result, our unbalanced classifier became even more unbalanced. Also, for such tasks, not a simple convolutional network with several layers is more suitable, but pre-trained large networks like Unet. 
