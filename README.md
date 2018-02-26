**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/calibration_original.jpg 
[image2]: ./examples/calibration_undist.jpg
[image3]: ./examples/pipeline_original.jpg 
[image4]: ./examples/pipeline_undist.jpg 
[image5]: ./examples/pipeline_binary_threshold.jpg 
[image6]: ./examples/pipeline_perspective.jpg 
[image7]: ./examples/pipeline_result.jpg 
[image8]: ./examples/WarpSRC.jpg 

[video1]: ./project_video.mp4 "Video"
[video2]: ./project_video_output.mp4 "Result Video"


---

### Camera Calibration

#### 1. How I computed the camera matrix and distortion coefficients.

I start by counting the corner numbers in the calibration images, there are 9X6 corners in each image. After that, I created "object points", which are (x, y, z) coordinates with the same shape of the corners(9X6).

I read in all the images in ./camera_cal, and get the pixel coordinates of corners using cv2.findChessboardCorners(). With these image corner points and object points, I am able to get the image calibration matrix by using cv2.calibrateCamera() and can use this on calibration the camera distortion later by using cv2.undistort(). 
(In line 7-40 of LaneFinding.py)

There are some example images for camera calibration, the left one is the first image and the second one is the image after calibration.
![alt text][image1]![alt text][image2]

### Pipeline

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

After applying
```python
undist = cv2.undistort(img,mtx,dist,None,mtx)
```
The undistorted output will be like the following image:
![alt text][image4]

#### 2. Color transforms and gradients thresholded binary image. 

First, I transformed the image to HLS color space, and use the L channel and S channel for my thresholding condition.

Secondly, I transformed my image to gray scale and apply cv2.Sobel() to calculate the gradient of the image. I used the 2-norm of x-direction sobel and y-direction sobel for my threshold.

Then I combine these conditions, with  the condition: (L_channel binary) and [(S_channel binary) or (2-norm of gradient binary)]

Here's an example of my output for this step. 
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
