
## Submission Template

### Project overview

This is a repository containing the details of the project Udacity Self Driving Car, where we are using Tf object detection API for better detetion of objects such as cars,pedestrians, cyclists etc. This repository contains the details to download the tfrecord sample files from Cloud storage and then split them for training purposes on the object detection API. The dataset used for this purpose is [Waymo](https://waymo.com/open/) which can be downloaded from the [Google Cloud Storage Bucket]((https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/). In this case, we will be using tfrecord files which we will be modified into tf.Train.Example for the object detection api format. We will also be splitting the dataset into training, validation and testing sets using np.split in  "create_splits.py" python program.  

### Set up

As mentioned in the project rubrics, GPU compatible system should be present for this.

- First the project files should be downloaded through git clone from [this repository](https://github.com/udacity/nd013-c1-vision-starter)
- Navigate to the root directory of the project and use the docker file and requirements.txt from the "build" directory
- The following command should be run from inside the "build" directory:

```
docker build -t project-dev -f Dockerfile.gpu
```

- Then we create a docker container to run the created image.
- Inside the container, we can use the gsutil command  to download the tfrecord from cloud storage:

```
curl https://sdk.cloud.google.com | bash
```
-Authentication can be done using 

```
gcloud auth login

```
- The following libraries can be installed

```
pip install tensorflow-gpu==2.3.0
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

### Dataset

In the dataset, we have to fit rectangular bounding boxes on the images with objects ,which includes pedestrians, cyclists and cars.Images are taken from different places, and different weather conditions and at different time of the day (day/night).The image set contains diverse set of images of which some are blurry, clear, light and some are dark. A sample image in dark and foggy background is provided below

<img src="https://i.imgur.com/c3F958n.png">

<img src="https://i.imgur.com/TJ8Ea3C.png">



#### Dataset analysis

The dataset presents most of the labels being associated with cars and pedestrians with the sample size of cyclists being very small. The proportionate amount of the counts of the different labels are in "Exploratory Data Analysis.ipynb" file. There is a class imbalance which can be handled with oversampling techniques. A sample image of the proportional counts of the labels (cars,pedestrians,cyclists) are shown below:

<img src="https://i.imgur.com/8jtDAmz.png">

Images are taken in different environments(subway/highway/city) with different weather conditions(foggy/sunny) and different times of the day(day/night).The bounding boxes are red for the vehicles, green for the cyclists and blue for the pedestrians.

![img1](images from Exploratory data analysis/img1.png)![img2](images from Exploratory data analysis/img2.png)![img3](images from Exploratory data analysis/img3.png)![img4](images from Exploratory data analysis/img4.png)![img5](images from Exploratory data analysis from Exploratory data analysis/img5.png)![img6](images from Exploratory data analysis/img6.png)![img7](images from Exploratory data analysis/img7.png)![img8](v/img8.png)![img9](images from Exploratory data analysis/img9.png)![img10](images from Exploratory data analysis/img10.png)

The analysis is updated in the "Exploratory Data Analysis.ipynb" notebook.
Further analysis have been done in the "Explore augmentations.ipynb" notebook. Hereby, I will attach some images from the database, under different circumstances:

![img1](images from Explore augmentations/img1.png)![img2](images from Explore augmentations/img2.png)![img3](images from Explore augmentations/img3.png)![img4](images from Explore augmentations/img4.png)![img5](images from Explore augmentations/img5.png)![img6](images from Explore augmentations/img6.png)![img7](images from Explore augmentations/img7.png)![img8](images from Explore augmentations/img8.png)
#### Cross validation

Using 100 tfrecord files, first shuffle the data randomly and then split into training,testing and validation sets. The reason for random shuffling is to
reduce the class imbalance in each sample. 

In this case, we are using 0.75 : 0.15 as the proportion of training and validation data since we are using only 100 tfrecord samples. This ensures that we have sufficient data for training as well as validation.We are using 10% (0.1) of the sample as the test set to check the error rate and if the model is overfitting.Ideally,overfitting should not be an issue since 75% is under training and rest 10% is for testing.

### Training 

#### Reference experiment

The residual network model ([Resnet](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)) without augmentation , model loss is shown below:

<img src="https://i.imgur.com/vC9whPX.jpg">

Initially the model was overffiting as the training loss was diverging from the validation loss.The training loss is indicated in orange and the validation loss in blue.This divergence indicates a significant error rate during model validation- an indication that the model is overfitting.
The precision and recall curves indicate that the performance of the model slowly increases, as both precision and recall start to increase. A high recall rate is often not suitable and the model performance is not that great.
Precision:

<img src="https://i.imgur.com/z4hSFrv.jpg">

Recall:

<img src="https://i.imgur.com/e3rRSdH.jpg">


#### Improve on the reference

To improve on the model performance, the first step was to augment the images by converting them to grayscale with a probability of 0.2. After this, we have clamped the contrast values between 0.6 and 1.0 such that more lighting datapoints are available for classification. A greater part of the images were a bit darker and increasing the brightness to 0.3 provided an even datapoint which could be better classified with the model.The pipeline changes are there in ```pipeline_new.config```

Augmentations applied:

- 0.02 probability of grayscale conversion
- brightness adjusted to 0.3
- contrast values between 0.6 and 1.0

Grayscale images:

<img src="https://i.imgur.com/ft9s4xx.png">

Night(Darker) Images:

<img src="https://i.imgur.com/rnhigAX.png">

Contrast Images:

<img src="https://i.imgur.com/pGJiRg3.png">



The details of the run can be found here : "Explore augmentations.ipynb"

The model loss with augmentation :

<img src="https://i.imgur.com/H4DtUd8.jpg">

Precision with Augmentation:

<img src="https://i.imgur.com/2aGRa93.jpg">

Recall with Augmentation:

<img src="https://i.imgur.com/wtTj62o.jpg">

The loss is lower than the previous loss (un-augmented model). This is an indication of better performance. There should be more samples of augmented datapoints such as
combining the contrast values with grayscale. Brightness can also be clamped within a limit instead of fixing it to 0.3
However the most important point is to add more samples of cyclists,pedestrians which are in a low quantity in the dataset. This is an inherent requirement since model biases play an important role in the loss curves and lesser the diversity in training samples, the lower will be the accuracy. 

We have reduced overfitting to an extent with augmentation, however better classification results would be resulting from a more balanced dataset.
