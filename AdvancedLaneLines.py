import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from os import listdir

def cameraCalibration(calib_img_folder):
	calib_img_names = []
	for img_name in listdir(calib_img_folder):
		calib_img_names.append(calib_img_folder+"/"+img_name)

	nx = 9
	ny = 6
	imgpoints = []
	objpoints = []

	objp = np.zeros((nx*ny,3),np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


	for fname in calib_img_names:
		img = cv2.imread(fname)
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)

	ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
	
	# for example images
	img = cv2.imread("test_images/test1.jpg")
	undist = cv2.undistort(img,mtx,dist,None,mtx)
	cv2.imwrite("examples/calibration_undist.jpg",undist)
	cv2.imwrite("examples/calibration_original.jpg",img)

	return mtx,dist

def getMyWarpMatrix(demo_img = None):
	# use some hard-coded coordinates for perspective transform
	# should we do it automaticly?
	src = np.float32(
		[
		[80,684],
		[1200,684],
		[744,463],
		[548,463]])

	
	dst = np.float32(
		[
		[140,670],
		[1180,670],
		[1180,50],
		[140,50]])

	if not demo_img is None:
		img = demo_img.copy()
		for i in range(src.shape[0]):
			pt1 = (src[i-1][0],src[i-1][1])
			pt2 = (src[i][0],src[i][1])
			cv2.line(img,pt1,pt2,(0,0,255),10)
		cv2.imwrite("examples/WarpSRC.jpg",img)

	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	return M,Minv
	
def binary_threshold(img):
	sobel_kernel=15
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]   

	l_thres = [120,255]
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel > l_thres[0]) & (l_channel <= l_thres[1])] = 255

	s_thres = [75,255]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > s_thres[0]) & (s_channel <= s_thres[1])] = 255

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	abs_sobel = np.sqrt(sobelx**2 + sobely**2)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	sobel_thres = [40,255]
	sobel_binary = np.zeros_like(abs_sobel)
	sobel_binary[(scaled_sobel > sobel_thres[0]) & (scaled_sobel <= sobel_thres[1])] = 255

	combined = np.zeros_like(s_channel)
	combined[(l_binary == 255) & ((s_binary == 255) | (sobel_binary == 255))] = 255

	return combined
# Reference: Lesson15, Section 33: Finding the Lines
def find_lanes(binary_warped,demo = False): 
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	if len(lefty) <=2 or len(righty) <=2:
		return None,None,None

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	if demo:
		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
									  ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
									  ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.imsave("examples/findLanes.png",result)
	# Measure Radius of Curvature for each lane line
	ym_per_pix = 30.0/720.0 # meters per pixel in y dimension
	xm_per_pix = 3.7/700.0 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space

	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	y_eval = np.max(ploty)

	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


	R = (left_curverad + right_curverad)/2
	right_crossing, left_crossing = right_fitx[-1], left_fitx[-1]
	#center should be 1280/2
	center = binary_warped.shape[1]/2
	offset = (center - ((right_crossing+left_crossing)/2.0)) * xm_per_pix

	return left_fitx, right_fitx,ploty, R, offset

def pipeline(img,demo=False):

	undist = cv2.undistort(img,mtx,dist,None,mtx)
	binary = binary_threshold(undist)
	warped = cv2.warpPerspective(binary,M,img.shape[0:2][::-1])

	lanes = np.zeros(img.shape,dtype='uint8')
	left_fitx, right_fitx,ploty, R, offset = find_lanes(warped,demo)
	if left_fitx is None:
		return img

	pt_left,pt_right = [],[]
	for i in range(len(ploty)):
		pt1 = [int(left_fitx[i]),int(ploty[i])]
		pt2 = [int(right_fitx[i]),int(ploty[i])]
		pt_left.append(pt1)
		pt_right.append(pt2)


	pt_left = np.array(pt_left,dtype=np.int32)
	pt_right = np.array(pt_right,dtype=np.int32)
	line_points = np.vstack((pt_left, pt_right[::-1]))
	cv2.polylines(lanes, [pt_left], isClosed=False,color=(255,0, 0),thickness=10)
	cv2.polylines(lanes, [pt_right], isClosed=False,color=(255,0, 0),thickness=10)
	cv2.fillPoly(lanes, [line_points],(0,255, 0))

	lanes = cv2.warpPerspective(lanes,Minv,lanes.shape[0:2][::-1])

	result = cv2.addWeighted(undist, 1, lanes, 0.5, 0)

	position = "right"
	if offset < 0:
		position = "left"
	cv2.putText(result, 'Offset from Center: {:.2f}m {} from center'.format(np.abs(offset),position), (80,80),
		fontFace = 16, fontScale = 1, color=(0,0,255), thickness = 2)
	cv2.putText(result, 'Radius of Road Curvature: {:.2f}(m)'.format(R), (80,160),
		fontFace = 16, fontScale = 1, color=(0,0,255), thickness = 2)
	if demo:
		cv2.imwrite("examples/pipeline_original.jpg",img)
		cv2.imwrite("examples/pipeline_undist.jpg",undist)
		cv2.imwrite("examples/pipeline_binary_threshold.jpg",binary)
		cv2.imwrite("examples/pipeline_perspective.jpg",warped)
		cv2.imwrite("examples/pipeline_result.jpg",result)

	return result

test_images = []

for img_name in listdir("test_images"):
	test_img = cv2.imread("test_images/"+img_name)
	test_images.append((img_name,test_img))

M, Minv = getMyWarpMatrix(test_images[0][1])
mtx,dist = cameraCalibration("camera_cal")

pipeline(test_images[0][1],demo = True)

for img in test_images:
	output = pipeline(img[1])
	cv2.imwrite("output_images/"+img[0],output)

output = "project_video_output.mp4"
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(output, audio=False)