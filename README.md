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
[image9]: ./examples/findLanes.jpg


---

### Camera Calibration

#### 1. How I computed the camera matrix and distortion coefficients.

I start by counting the corner numbers in the calibration images, there are 9X6 corners in each image. After that, I created "object points", which are (x, y, z) coordinates with the same shape of the corners(9X6).

I read in all the images in ./camera_cal, and get the pixel coordinates of corners using cv2.findChessboardCorners(). With these image corner points and object points, I am able to get the image calibration matrix by using cv2.calibrateCamera() and can use this on calibration the camera distortion later by using cv2.undistort(). 
(In line 7-40 of AdvancedLaneLines.py)

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
(Located in line 73-99 of AdvancedLaneLines.py)

Here's an example of my output for this step. 
![alt text][image5]

#### 3. Perspective transform.

First, I start with choosing my area of interest in the camera image.
![alt text][image8]

This four corners of this area is :
```python
src = np.float32(
	[
	[80,684],
	[1200,684],
	[744,463],
	[548,463]])
 ```

And I defined a destination area for the warping projection:
```python
dst = np.float32(
	[
	[140,670],
	[1180,670],
	[1180,50],
	[140,50]])
 ```

With the source and destination of warping defined we can warp the image to "Bird-eye's view" with the opencv getPerspectiveTransform function:
```python
M = cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(binary,M,imshape) # binary is the output of the last session
```

(Located in line 42-71 of AdvancedLaneLines.py)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Identifying lane-line pixels and fit their positions with a polynomial

With the techniques mentioned in class(Lesson15, Section 33: Finding the Lines), we are able to find the lane lines.

First we divide the image into strides in y-direction, for each stride we perform a sliding window adding up the nonzero pixels within the window.

Secondly, we can make histogram to identify two strongest peaks and assume they to be correspond to the two lane lines. 

After get all the left and right lane points on each strides, we can perform second degree(quadratic) polynomial fit to find the left and right lane lines.

(Located in line 100-231 of AdvancedLaneLines.py)

Here is the example of my polynomial fit for the image:

![alt text][image9]

#### 5. How to calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

Since we already use a second degree polynomial fit for our lanes, it can be expressed as:

"f(y)=Ay^2+By+C"

And there is a closed form for curvature of a quadratic line:

Radius of curve = (1+(2Ay+B)^2) ^1.5 / |2A|; curvature is the reciprocal of radius.

Given this we can calculate the curvature of the lanes at y meters ahead.
(Source: https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

(Located in line 210-229 of AdvancedLaneLines.py)

#### 6. Example image of my result.

I implemented this step in lines # through # in my code. Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Link to your final video output.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my pipeline, if there's less than 3 pixels detected after binary thresholding, it may have problem to fit a second degree polynomial line and it might fail. We can detect this and prevent doing polynomial fit with too feww data points.

Besides, in very bad(too strong or too weak) or uneven lighting condition, such as conditions in challenge videos, the binary threshoolding has problem to fetch the lane pixels. We could improve this by tracking the lanes by considering previous processed frames. In this project, we never consider the correlations between consecutive frames, and this should help since we can fairly assume the correlation of lane position between consecutive frames should be vaery high in real world.