# Project Name: **Optical Character Recogintion**:
#### A powerful open-source Reinforcement learning model for detecting characters present in the image.
This repository holds the code and resources for **OCR**, a machine learning model trained to perform **optical text recogonistion** with exceptional accuracy. Built using `tensorflow`,`numpy`,`pickle`,`matplotlib`,`os`, the model excels at **recogonising handwritten letters**, making it ideal for applications such as search by image Eg. ****Google Images****. 
The algorithm takes an **image containing words as input** and **outputs the detected words**.
Optionally, the words are sorted according to reading order (top to bottom, left to right).

![example](/outputv1.png)


## Installation
* Hit the following commands in your terminal:-
  
```
pip install numpy
pip install tensorflow
pip install pickle
pip install matplotlib
```

## Usage
Clone the repository in your local machine,put the image that you want to predict in test_images folder and ensure that it should be in `.png` format.
If you want to compile the model then start the training it should take around (~1.3hrs) to train the model as the data set is qiuite huge so to avoid that i also have a pretrained
model saved in the repos named `ocr_model_50_epoch.h5`load it via -

```
custom_objects = {"CTCLayer": CTCLayer}

reconstructed_model = keras.models.load_model("./ocr_model_50_epoch.h5", custom_objects=custom_objects)

prediction_model = keras.models.Model(
  reconstructed_model.get_layer(name="image").input, reconstructed_model.get_layer(name="dense2").output
)
```


## Algorithm

The illustration below shows how the algorithm works:

* top left: input image
* top right: apply filter to the image
* bottom left: threshold filtered image
* bottom right: compute bounding boxes

The filter kernel with size=25, sigma=5 and theta=3 is shown below on the left. 
It models the typical shape of a word, with the width larger than the height (in this case by a factor of 3). 
On the right the frequency response is shown (DFT of size 100x100). 
The filter is in fact a low-pass, with different cut-off frequencies in x and y direction.

## Output sample

![Screenshot](/outputv1.png)


## How to select parameters

* The algorithm is **not scale-invariant**
    * The default parameters give good results for a text height of 25-50 pixels
    * If working with lines, resize the image to 50 pixels height
    * If working with pages, resize the image so that the words have a height of 25-50 pixels
* The sigma parameter controls the width of the Gaussian function (standard deviation) along the x-direction. Small
  values might lead to multiply detection per word (over-segmentation), while large values might lead to a detection
  containing multiple words (under-segmentation)
* The kernel size depends on the sigma parameter and should be chosen large enough to contain as much of the non-zero
  kernel values as possible
* The average aspect ratio (width/height) of the words to be detected is a good initial guess for the theta parameter

The best way to find the optimal parameters is to use a dataset (e.g. IAM) and optimize the parameters w.r.t. some
evaluation metric (e.g. intersection over union).

