# UNet-lesion-detection
In this project U-net architecture is applied to segment lesions on CT scans caused by Coronavirus. U-net architecture is adopted because it is specifically designed for medical image segmentation. 

# Lesion Detection on Lung CT (Computed Tomography) Scans for

# Coronavirus Infections

```
Berat Tuna KARLI
Graduate School of Informatics, Middle East Technical University, Ankara, Turkey
```
## Problem definition

Coronavirus pandemic has been affecting our lives in the global scale. Still, there is no ultimate cure for
the disease and the spread of the virus could not be brought under control, yet. Therefore, it still threatens
our health and affecting our lives in so many ways. Coronavirus causes several types of lesions in human
body, especially in lung. Lesions are briefly the deformation caused by corresponding disease and lesion
detection is the process of detecting lesions. In chest CT (Computed Tomography) scans; ground-glass,
consolidation, pleural effusion or crazy paving patterns could be used to detect COVID-19 infections [1].

Currently, more and more laboratory and people focus in medical image detection. With the Coronavirus
pandemic, lesion detection on lung become prominent in medical image detection. Lesion detection
would provide precise knowledge about the size and location lesions. Manual and traditional medical
testings are costly considering its labor and time cost. Nowadays, detection of lesions in CT automatically
is an integral assignment to paramount importance of clinical diagnosis [2].

This paper proposes a project to detect lesions, caused by Coronavirus, on Lung CT scan with Deep
Learning. Different than previous CT studies, this project focuses on lesion detection caused by only
coronavirus. Also, different than current Coronavirus studies, this report will specifically focus on Lung CT
scans and be targeted to detect lesions using CT scans. Thus, this project aims to detect lesions
automatically on Lung CT scans of Covid-19 patients in order to minimize medical labor and time cost
especially during the pandemic.

## Literature review

There are several studies to detect lesions on CT scan prior to Coronavirus. The paper DeepLesion:
Automated Mining of Large-scale Lesion Annotations and Universal Lesion Detection with Deep Learning
[3] presents very comprehensive study on lesion detection for various regions (i.e. liver, lung, kidney,
pelvis, etc.). However, DeepLesion focuses on many regions and it is specific to cancer patients not
Coronavirus patients.

There are not many papers on lesion detection for Coronavirus patients since data is limited and the virus
is discovered only few months ago. There is a study to detect Coronavirus from CT scan using deep-
learning techniques [4]. However, in this article lesion annotation is not applied for CT scans. Thus, the
algorithm could only detect coronavirus exists or not without detecting lesion location and type in CT
scan.

There are numerous effective studies for object detection (e.g. AlexNet [ 5 ], ResNet [ 6 ]), However,
considering that the purpose of this paper is to detect lesions pixel-wise, it is better to use image
segmentation methods (e.g. semantic segmentation) instead of regular object detection algorithms with
bounding box. In this purpose, there is a convolutional architecture specifically designed for biomedical
image segmentation published at University of Freiburg [ 7 ]. Basically, this architecture encodes input
image and decode segmentation map of the image as output as shown in figure 1.

![image](https://user-images.githubusercontent.com/29654044/124667034-84016f00-deb7-11eb-95fa-281b1b358aab.png)

```
Fig. 1. U-net architecture [ 7 ].
```
#### Advantages of U-net:

- The U-Net architecture combines information of location which comes from contracting path with
    the contextual information in the expanding path to lastly acquire over-all knowledge of context
    and localization, this is essential to generate decent segmentation map [ 8 ].
- There are no dense layer, therefore scans of different sizes could be fed as input (considering the
    only parameters to learn on convolution layers are the kernel, and the size of the kernel is
    independent from input image’ size) [ 8 ].
- The use of massive data augmentation is significant in areas (i.e. biomedical object detection-
    segmentation), as the number of annotated samples is generally restricted [ 8 ].

## Dataset

Dataset includes COVID-19 CT scans from the Italian Society of Medical and Interventional
Radiology’s collection [ 9 ]. There are scans with size 512x512 and segmentations (masks) with again
size 512x512.

![image](https://user-images.githubusercontent.com/29654044/124667048-895eb980-deb7-11eb-8309-2e75ba1cff90.png)

```
Fig. 2. Segmented COVID-19 CT axial slice. Ground-glass opacities in blue, consolidation in yellow and pleural effusion in green.
```

Dataset contains 100 axial CT images from with COVID-19 gathered from nearly 60 patients. Images
in dataset were segmented by radiologists using 3 labels: ground-glass (mask value =1),
consolidation (=2) and pleural effusion (=3). Images are segmented with mask using different colors
as show in figure 2.

Complete dataset involves following data:

- Training images as .nii.gz (151.8 Mb) – 100 slices
- Training masks as .nii.gz (1.4 Mb) – 100 masks
- CSV file connecting slice no. with SIRM case no. (0.001 Mb)
- Test images as .nii.gz (14.2 Mb) – 10 slices

In case more information is needed for the further studies, every image in the dataset contains publicly
accessible detailed information about patient age, sex, comorbidity condition and other symptoms [ 10 ].

## Methodology

In this project U-net architecture is applied to segment lesions on CT scans caused by Coronavirus. U-net
architecture is adopted because it is specifically designed for medical image segmentation. One sample of
input data and mask data are displayed in figure 3. All 3 types of masks of the sample input are displayed
in figure 4.

![image](https://user-images.githubusercontent.com/29654044/124667062-8fed3100-deb7-11eb-8ce4-2ad70706ee9f.png)

```
Fig. 3. Sample training and mask data.
```

![image](https://user-images.githubusercontent.com/29654044/124667072-94b1e500-deb7-11eb-800d-6aee61d09bb0.png)

```
Fig. 4. All mask types of sample data.
```

However, U-net architecture is designed to detect single object. However, dataset contains 3 different
types of lesion masks. Thus, 2 different methods are applied to modify U-net architecture to current
problem.

#### First Method

First method is simply to reduce lesion types to 1, thus object types to binary 2 types (i.e. lesion or not
lesion). All types of masks (i.e. Ground-glass, consolidation and pleural effusion) are accepted as regular
mask and labelled identically. After this pre-process, mask data is binarized to 0 and 1. Grey-scaled input
data and mask data are shown in figure 5.

In the first method, architecture did not require additional adaption. The output dimension of the model
is (None, 512, 512, 1 ). Sigmoid activation function is applied for the output layer. The loss function is set
to Binary Cross-Entropy and Rmsprop optimizer is applied.

![image](https://user-images.githubusercontent.com/29654044/124667087-9b405c80-deb7-11eb-8220-33ffc5cd83e0.png)


```
Fig. 5. Sample Preprocessed input CT scan and label (binarized mask) for the first method.
```
#### Second method

Second method was to adapt U-net architecture into multi (i.e. categorical) object segmentation. For this
purpose, initially the input is converted into one-hot-encoding input. This edition added one more
dimension to input with length of 4 as the number of object types (i.e. 3 lesion types mentioned above
plus non-lesion type). The size of input for one batch has updated to 20x512x512x 4. After this pre-process,
mask data are shown in figure 6.

![image](https://user-images.githubusercontent.com/29654044/124667100-9ed3e380-deb7-11eb-9976-2bcb52cfdef1.png)


```
Fig. 6. Sample Preprocessed input CT scan and decoded one-hot-encoded label for the second method.
```

Also, architecture requires adaption to this edition in second method. Thus, output dimension of the
model is changed from (None, 512, 512, 1 ) to (None, 512, 512, 4 ). Activation function is changed to
Softmax from Sigmoid. Because the output now contains 4 different values representing every category
for each pixel, thus probabilities between 1 and 0 should be calculated among these 4 object types. Also,
the loss function is adapted to Categorical Cross-Entropy from Binary Cross-Entropy, the reason is to find
output neuron with the highest probability which derives from softmax. The optimizer is changed from
Rmsprop to Adam optimizer. Rmsprop optimizer could not converge and loss function did not change,
thus Adam Optimizer is set for this method.

#### The Architecture

In both methods, data is split into training (80%) and test (20%) sets. Thus, training data contains 80
images and test data contains 20 images. Final training batch size is set to 32 and testing batch size is set
to 20. Epoch number is set to 300.

In both methods, U-net architecture is used. In the architecture design there are 4 main parts; down-
sampling, bottleneck, up-sampling and classification. First 3 parts are same in both methods, but
classification part has structural differences among the two methods:

#### 1. Down-sampling

```
This part is composed of 6 blocks. Each block is composed of;
```
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding).
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding).
- 2x2 Max Pooling.

#### 2. Bottleneck

```
This part of the architecture is between the down-sampling and up-sampling paths. The bottleneck is
built from simply 2 convolutional layers (with ‘relu’ activation and ‘same’ padding).
```
#### 3. Up-sampling

```
This is the expanding path which is composed of 5 blocks. Each of these blocks is composed of;
```
- Deconvolution layer with stride 2.
- Concatenation with the corresponding cropped feature map from the down-sampling path.
- 3x3 Convolution layer + activation function (with ‘relu’ activation and ‘same’ padding).
- 3x3 Convolution layer + activation function (with ‘relu’ activation and ‘same’ padding).

#### 4. Classification

```
a. First Method
First method contains;
```
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding).
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding).
- Final Convolution Layer (with ‘sigmoid’ activation).
- Model compiled with Rmsprop optimizer function.
- Model compiled with Binary Cross-Entropy loss function.


```
b. Second Method
Second method contains;
```
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding)
- 3x3 Convolution Layer + activation function (with ‘relu’ activation and ‘same’ padding)
- Final Convolution Layer (with ‘sofmax’ activation)
- Model compiled with Adam optimizer function.
- Model compiled with Categorical Cross-Entropy loss function.

For the loss metric, Categorical Cross-Entropy is used for the first Method and Categorical Cross-Entropy
is used for the second Method.

For the accuracy metric, the intersection over union (IoU) metric is applied to evaluate accuracy. IoU is
the most popular evaluation metric used in the object detection benchmarks to measure the performance
of object detection and segmentation algorithms [ 11 ]. IoU is simply the ratio of (ytrue ∩ ypred) divided
by (ytrue ∪ ypred).

Since there are 2 different methods, 2 different IoU metric function are adapted. For the first method, IoU
is applied for binary labels for each pixel in 512x512 images. For the second method, 4 different IoU are
calculated for each object type (considering one-hot-encoded label and output), and mean of these 4
different IoU is calculated for final IoU metric.

## Result

Google Colab is used to train the models considering it has higher GPU performance compared to personal
computer. Python 3 is used in Google Colab and the GPUs available in Colab includes Nvidia K80s, T4s, P4s
and P100s [ 12 ]. Tensorflow2/Keras used as the main library.

#### First Method Results

The first method is designed to detect any kinds of lesions in a binarized way. Thus, it is indeed,
appropriate for U-net architecture nature directly. Also, the complexity of the problem is reduced in
dimensionality. Therefore, first model is expected to have better results in less time.

The training time of the first model is 31.48 minutes for 300 epochs with training batch size of 32 and test
batch size of 20.

The loss graph for the model seems extremely stable, it could be seen in the figure 8. At the first trials the
model predicts all pixels as non-lesions. However after some iterations, the model started to decode some


pixels as lesions and it gets better predictions. In figure 9 , 3 different prediction results are displayed at
29 th, 50th and 192nd epochs to observe prediction accuracy visually.

![image](https://user-images.githubusercontent.com/29654044/124667193-bad78500-deb7-11eb-8462-e28cd8cab928.png)

```
Fig. 8. Loss graph for train and test data according to epoch number for the first method.
```

![image](https://user-images.githubusercontent.com/29654044/124667208-c165fc80-deb7-11eb-904f-8c5c4838bb50.png)
![image](https://user-images.githubusercontent.com/29654044/124667214-c32fc000-deb7-11eb-93fa-e8a1d152cbef.png)
![image](https://user-images.githubusercontent.com/29654044/124667220-c4f98380-deb7-11eb-8d5a-2f0f89b0dec8.png)

```
Fig. 9. Top-left figure is 29th epoch, top-right figure is 50th epoch and figure in the bottom is 192nd epoch predictions.
```
For the accuracy measure as mentioned in Methodology part, IoU metric is used. This metric shows that
there is overfitting. Because of the accuracy fluctuation, the breaking point could not be precisely
understood but it could be said that there is overfitting after 200 th- 250 th epochs. While training accuracy
is constantly increasing, test accuracy started to decrease after 200 th- 250 th epochs. This could be seen in
the figure 10.

Consequently, it would be better not to train the first model more than 250 epochs to prevent the model
from overfitting.

![image](https://user-images.githubusercontent.com/29654044/124667230-ca56ce00-deb7-11eb-8d67-dbb9666a7cc3.png)

```
Fig. 10. Accuracy graph for train and test data by epoch number for the first method.
```
There is not completely irrelevant predictions after 250th epoch but, there are several missing predictions
specifically when the segmented region is narrow. The prediction after 264 th epoch is displayed in figure
11 to display missing predictions in narrow regions.

![image](https://user-images.githubusercontent.com/29654044/124667241-cdea5500-deb7-11eb-8efa-8ab47c76c085.png)

```
Fig. 11. Prediction after 264 th epoch. Fine lines and some narrow regions could not be segmented as lesions.
```
#### Second Method Results

The second method is designed specifically to detect all kinds of lesions which is more comprehensive
classification algorithm. So, this method was not originally suitable for U-net architecture. Therefore, it
required additional adaptations mentioned in methodology part. Also, the complexity of the problem is
increased in dimensionality considering one-hot-encoded input and labels. So, second model is expected
to have longer training time and less accuracy according to increased complexity.

The training time of the second model is 42.48 minutes for 300 epochs with training batch size of 32 and
test batch size of 20.

The loss graph for the second model is displayed in the figure 11. It is seen that while training loss is
constantly decreasing with a slight slope, the test loss started to increase after nearly 150th epoch. This
shows that the model is overfitting after 150 th epoch approximately.


At the first trials the model predicts all pixels as non-lesions as the first model and after some iterations,
the second model also started to decode some pixels as lesions and it gets better predictions. In figure 12 ,
3 different prediction results are displayed at 29 th, 50th and 192nd epochs to observe prediction accuracy
visually.

Consequently, it would be better not to train the second model more than 150 epochs to prevent the
model from overfitting.

![image](https://user-images.githubusercontent.com/29654044/124667275-da6ead80-deb7-11eb-99a7-e33c91dd68b2.png)

```
Fig. 12. Loss graph for train and test data by epoch number for the second method.
```

![image](https://user-images.githubusercontent.com/29654044/124667280-de023480-deb7-11eb-83e8-74a4a116207b.png)
![image](https://user-images.githubusercontent.com/29654044/124667293-e0648e80-deb7-11eb-9ab0-5e8e52b29cfa.png)
![image](https://user-images.githubusercontent.com/29654044/124667296-e22e5200-deb7-11eb-9a84-2d94a068f786.png)

```
Fig. 13. Top-left figure is 2 3 th epoch, top-right figure is 83 th epoch and figure in the bottom is 97 nd epoch predictions.
```

For the accuracy measure as mentioned in Methodology part, IoU metric is redesigned according to one-
hot-encoded input-output for second method. However, this metric could not perform well with multi
object segmentation. At the first stages, the non-lesion predictions increase the accuracy, but when the
model started to predict some lesions after 80th epoch approxiamtely, the accuracy started to constantly
increase with a slight slope.

![image](https://user-images.githubusercontent.com/29654044/124667332-efe3d780-deb7-11eb-8579-1c5105d441eb.png)

```
Fig. 14. Accuracy graph for train and test data by epoch number for the second method.
```
The model could be considered successful according to detect shapes of lesions. However, sometimes it
has complete misclassifications among all segmented lesions. Especially after 150th epoch where the
overfitting started, the predicted lesion types could be completely or partially misclassified. An example
is shown in figure 14.

![image](https://user-images.githubusercontent.com/29654044/124667354-f5d9b880-deb7-11eb-90d2-fa3fb5891ca5.png)

```
Fig. 15. Prediction after 155 th epoch. Almost all should be segmented as yellow, but mostly segmentation is predicted in green.
```

## Conclusion

In conclusion, this project is targeted to detect lesions which are caused by Coronavirus on Lung CT scan.
This project focuses on lesion detection caused by only coronavirus, thus only lung scans of coronavirus
patients are used in this purpose. There is no proven and mature research on Lung CT scans for
Coronavirus patients which is aimed to detect lesions on CT scans. Thus, this project aims to assist medical
personnel by automatically detecting lesions on Lung CT scans of Covid-19 patients to reduce labor and
time cost in the pandemic.

U-net architecture is adapted as the main deep neural network architecture which is specifically designed
for medical image segmentation projects at the University of Freiburg. 2 different methods are applied,
thus 2 different models are generated. First Method is using pixel-wise binary classification to detect
lesion pixels and non-lesion pixels. In order to adapt data into desired format of first method, all mask
values are marked identically for true labels and predicted labels. Second model is used to detect lesions
and classify detected lesions. Input and predictions are transformed to one-hot-encoded labels, also
architecture of U-net is redesigned to adapt pixel-wise multi object classification which could be summed
as multi object segmentation.

First model acquired better loss and IoU metric results. However, it should be considered that first model
is naturally be applicable for U-net architecture and second model has higher complexity as it could be
also seen from training time of models.

Considering that both models are trained with very limited dataset (i.e. 100 CT scans), the research could
be improved after more Coronavirus scans acquired. Also, platform, the hardware, could be improved
after gathering more scans (e.g. 10000 scans). Because Google Colab has limited training duration as 12
hours. Besides, limited dataset, U-net architecture could be customized to models in order to improve
performance.

Considering to research area, the scope of the project could be extended. In this research, only CT scans
of Coronavirus patients are used. Though, a model could be trained to first detect lesions and then classify
lesions whether they are caused by Coronavirus or regular pneumonia. This approach changed, indeed
improve the problem. The problem becomes more comprehensive and complicated. In order to train a
model to solve this particular problem, more detailed architecture could be designed or multiple network
architectures could be combined to resolve the problem (e.g. combination of ResNet with U-net for
enhanced deep convolutional neural networks in medical image segmentation [ 13 ]).


### References

[1] Radiology Assistants, COVID-19 Image findings, https://radiologyassistant.nl/chest/lk-jg- 1

[2] Zhang, Hui & Chen, Yurong & Song, Yanan & Xiong, Zhenlin & Yang, Yimin & Wu, Q. M. Jonathan.

(2019). Automatic Kidney Lesion Detection for CT Images Using Morphological Cascade Convolutional

Neural Networks. IEEE Access. PP. 1-1. 10.1109/ACCESS.2019.2924207.

[3] Yan K, Wang X, Lu L, Summers RM. DeepLesion: automated mining of large-scale lesion
annotations and universal lesion detection with deep learning. _J Med Imaging (Bellingham)_.
2018;5(3):036501. doi:10.1117/1.JMI.5.3.

[4] Zheng, C., Deng, X., Fu, Q., Zhou, Q., Feng, J., Ma, H., Wang, X. (2020). Deep Learning-based Detection
for COVID-19 from Chest CT using Weak Label. doi:10.1101/2020.03.12.

[ 5 ] Krizhevsky, A., Sutskever, I. and Hinton, G., 2017. ImageNet classification with deep convolutional
neural networks. Communications of the ACM, 60(6), pp.84- 90.

[ 6 ] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2016.

[ 7 ] Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image
Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and
Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol

9351. Springer, Cham

[ 8 ] U-Net¶. (n.d.). Retrieved June 28, 2020, from [http://deeplearning.net/tutorial/unet.html](http://deeplearning.net/tutorial/unet.html)

[ 9 ] MedSeg, COVID-19 CT segmentation dataset, [http://medicalsegmentation.com/covid19/](http://medicalsegmentation.com/covid19/)

[ 10 ] COVID-19: CASISTICA RADIOLOGICA ITALIANA, COVID- 19 database,
https://www.sirm.org/category/senza-categoria/covid-19/

[ 11 ] Rahman M.A., Wang Y. (2016) Optimizing Intersection-Over-Union in Deep Neural Networks for
Image Segmentation. In: Bebis G. et al. (eds) Advances in Visual Computing. ISVC 2016. Lecture Notes in
Computer Science, vol 10072. Springer, Cham

[ 12 ] Colaboratory. (n.d.). Retrieved June 28, 2020, from
https://research.google.com/colaboratory/faq.html

[ 13 ] Zhang Q., Cui Z., Niu X., Geng S., Qiao Y. (2017) Image Segmentation with Pyramid Dilated Convolution
Based on ResNet and U-Net. In: Liu D., Xie S., Li Y., Zhao D., El-Alfy ES. (eds) Neural Information Processing.
ICONIP 2017. Lecture Notes in Computer Science, vol 10635. Springer, Cham


