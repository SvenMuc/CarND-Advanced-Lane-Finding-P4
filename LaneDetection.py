import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('macosx', force=True)
#matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from enum import  Enum, unique
from Calibration import *
from CoreImageProcessing import *
import threading
import click


@unique
class Mode(Enum):
    """ Enum detection mode. """
    NO_DETECTION = 0                       # no line detected
    DETECTED = 1                           # line detected in previous frame
    NEW_LINE = 2                           # new line detected in current frame
    PREDICTED = 3                          # line from previous frame applied


class LaneDetection:
    """ Detects the ego lane (left and right line markings) from RGB images. """

    # enable debug for individual processing steps
    debug_preprocessing = False            # If true, plot pre-processing debug data

    calib = None                           # Camera calibration to undistort the images
    cip = None                             # Core image processing instance

    # -----------------------------------------------------------------------
    # Image pre-processing parameters

    # RGB channel thresholds
    threshold_r_supression_mode = False    # If true, R channel supression activated (usage of higher thresholds)
    threshold_r_sum_0 = 0.032              # hyteresis 0: percentage of red proportion (<= threshold_r_0, > threshold_r_1)
    threshold_r_sum_1 = 0.04               # hyteresis 1: percentage of red proportion (<= threshold_r_0, > threshold_r_1)
    threshold_r_sum_2 = 0.1                # hyteresis 2: percentage of red proportion (<= threshold_r_1, > threshold_r_2)
    threshold_r_0 = (190, 255)             # Threshold 0 for R channel [0..255] (usefull for yellow lines on dark surfaces)
    threshold_r_1 = (210, 255)             # Threshold 1 for R channel [0..255] (suppress bleeding on bright concrete)
    threshold_r_2 = (230, 255)             # Threshold 2 for R channel [0..255] (extremly suppress bleeding on bright concrete)
    threshold_g = (170, 255)               # Threshold for G channel [0..255]
    threshold_b = (185, 255)               # Threshold for B channel [0..255]

    # HLS channel thresholds
    threshold_h = (18, 25)                 # Threshold for H channel [0..179]
    threshold_l = (135, 255)               # Threshold for L channel [0..255]
    threshold_s = (100, 255)               # Threshold for S channel [0..255]

    # x-/y-sobel thresholds
    sobel_kernel_rgb = 13                  # Sobel kernel size, odd number [3..31] for RGB image
    threshold_r_gradient_x = (30, 170)     # Threshold for R channel x-gradients [0..255]
    threshold_b_gradient_x = (35, 150)     # Threshold for B channel x-gradients [0..255]

    sobel_kernel_hls = 15                  # Sobel kernel size, odd number [3..31] for HLS image
    threshold_s_gradient_x = (30, 170)     # Threshold for L channel x-gradients [0..255]

    # magnitude thresholds
    threshold_r_magnitude = (70, 200)      # Threshold for R channel magnitude of x-/y-gradients [0..255]
    threshold_b_magnitude = (70, 200)      # Threshold for B channel magnitude of x-/y-gradients [0..255]

    # -----------------------------------------------------------------------
    # Find lane lines parameters

    # warping rectangles
    warp_img_height = 720
    warp_img_width = 1280
    warp_src_pts = np.float32([[193, warp_img_height], [1117, warp_img_height], [686, 450], [594, 450]])
    warp_dst_pts = np.float32([[300, warp_img_height], [977, warp_img_height], [977, 0], [300, 0]])

    mode = Mode.NO_DETECTION               # detection mode
    max_prediction_cycles = 3              # max allowed number of frames in prediction mode
    prediction_count = 0                   # prediction counter

    prev_left_line_fit = None              # left polynomial line fit from previous frame
    prev_right_line_fit = None             # right polynomial line fit from previous frame
    prev_line_fit_y = None                 # y values of polynmial line fit from previous frame
    prev_line_fit_left_x = None            # x values of left polynmial line fit from previous frame
    prev_line_fit_right_x = None           # x values of right polynmial line fit from previous frame
    prev_left_curve_radius = None          # left curve radius in meter from previous frame
    prev_right_curve_radius = None         # right curve radius in meter from previous frame
    prev_avg_curve_radius = None           # average left/right curve radius in meter from previous frame
    prev_d_left = None                     # distance to left line in previous frame
    prev_d_right = None                    # distance to right line in previous frame

    # conversion parameters for warped image (birds-eye-view) to world space
    xm_per_pix = 3.7 / 652                 # meters per pixel in x dimension (lane width in US = 3.7 m)
    ym_per_pix = 3.0 / 80                  # meters per pixel in y dimension (dashed marker length in US = 3.0 m)
    cam_pos_x = 1280 / 2.                  # camera x-position in pixel (center of image)

    # -----------------------------------------------------------------------
    # Overlay setup

    overlay_ego_lane_enabled = True
    overlay_ego_lane_fits_enabled = True
    overlay_ego_lane_search_area_enabled = False

    # -----------------------------------------------------------------------
    # Images of all processing steps for debug purposes

    fig = None                                                  # Figure for debug plots
    debug_plot = []                                             # Array with axis (plots)
    img_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)          # input RGB image
    img_binary = np.zeros((720, 1280), dtype=np.uint8)          # pre-processing output binary image
    img_binary_warped = np.zeros((720, 1280), dtype=np.uint8)   # warped pre-pre-processing output binary image (birds-eye-view)
    img_line_search = np.zeros((720, 1280, 3), dtype=np.uint8)  # debug overlay for line detection algorithms

    img_binary_rlsb_3c = np.zeros((720, 1280, 3), dtype=np.uint8)       # colored RSB binary image
    img_binary_x_grad_rb_3c = np.zeros((720, 1280, 3), dtype=np.uint8)  # colored RSB x-gradient binary image
    img_binary_mag_rb_3c = np.zeros((720, 1280, 3), dtype=np.uint8)     # colored RSB magnitude binary image

    def __init__(self, calibration_file):
        """ Initialization method.

        :param calibration_file: Camera calibration file.
        """

        self.calib = Calibration(calibration_file)
        self.cip = CoreImageProcessing()
        self.init_debug_plots()

    def init_debug_plots(self):
        """ Initialize figure for debugging plots. """

        self.fig = plt.figure(figsize=(13, 7))
        self.fig.subplots_adjust(bottom=0.01, left=0.01, top=0.99, right=0.99, wspace=0.01, hspace=0.01)
        self.debug_plot.append(plt.imshow(self.img_rgb))
        plt.ion()
        plt.axis('off')
        # TODO: self.debug_plot.append(self.fig.add_axes([0.1, 0.8, 0.2, 0.2]))

    def update_debug_plots(self):
        """ Updates the debug plots with latest pipeline results. """

        img_rgb_overlay = self.img_rgb.copy()

        img = cv2.resize(self.img_binary_rlsb_3c, (320, 180))
        img_rgb_overlay[10:190, 10:330, :] = img

        img = cv2.resize(self.img_binary_x_grad_rb_3c, (320, 180))
        img_rgb_overlay[10:190, 340:660, :] = img

        img = cv2.resize(self.img_binary_mag_rb_3c, (320, 180))
        img_rgb_overlay[10:190, 670:990, :] = img

        img = cv2.resize(cv2.cvtColor(self.img_binary, cv2.COLOR_GRAY2RGB), (320, 180))
        img_rgb_overlay[200:380, 10:330, :] = img * 255

        img = cv2.resize(self.img_line_search, (320, 180))
        img_rgb_overlay[200:380, 340:660, :] = img

        # TODO: img_rgb_overlay[200:380, 670:990, :] = img

        cv2.putText(img_rgb_overlay, 'R(LS)B color', (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(img_rgb_overlay, 'RB x-grad', (342, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_rgb_overlay, 'RB Magnitude', (672, 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_rgb_overlay, 'Thresholded', (12, 214), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(img_rgb_overlay, 'Warped', (342, 214), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        # TODO: cv2.putText(img_rgb_overlay, 'Warped', (672, 214), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        self.debug_plot[0].set_data(img_rgb_overlay)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def preprocess_image(self, img_rgb):
        """ Pre-processing pipeline to prepare images for the lane line finding algorithms.

        Pipeline:
         - undistort RGB image
         - extract R, B and S channels
         - threshold each channel and create binary images
         - merge binary images

        :param img_rgb:       Input RGB image.

        :return: Returns a binary image with lane candidates. The binary image has the same dimensions as the input
                 image (pixel range = [0..1]).
        """
        img_rgb = self.calib.undistort(img_rgb)

        # separate RGB and HLS channels
        r_channel = img_rgb[:, :, 0]
        b_channel = img_rgb[:, :, 2]

        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        l_channel = img_hls[:, :, 1]
        s_channel = img_hls[:, :, 2]

        # threshold color channels
        img_binary_r = self.cip.threshold_image_channel(r_channel, self.threshold_r_0)
        img_binary_b = self.cip.threshold_image_channel(b_channel, self.threshold_b)
        img_binary_l = self.cip.threshold_image_channel(l_channel, self.threshold_l)
        img_binary_s = self.cip.threshold_image_channel(s_channel, self.threshold_s)

        img_binary_ls = np.zeros_like(img_binary_r)
        img_binary_ls[(img_binary_l == 1) & (img_binary_s == 1)] = 1

        # avoid red color channel bleeding (adjust thresholds)
        thres = 'T0'
        start_row = int(img_binary_r.shape[0]/1.65)
        nb_pixel = (img_binary_r.shape[0] - start_row) * img_binary_r.shape[1]
        r_sum = np.sum(img_binary_r[start_row:, :]) / nb_pixel

        if r_sum <= self.threshold_r_sum_0 and self.threshold_r_supression_mode:
            thres = 'T0'
            self.threshold_r_supression_mode = False
        elif r_sum >= self.threshold_r_sum_2:
            thres = 'T2'
            self.threshold_r_supression_mode = True
            img_binary_r = self.cip.threshold_image_channel(r_channel, self.threshold_r_2)
        elif r_sum >= self.threshold_r_sum_1 or self.threshold_r_supression_mode:
            thres = 'T1'
            self.threshold_r_supression_mode = True
            img_binary_r = self.cip.threshold_image_channel(r_channel, self.threshold_r_1)

        font = cv2.FONT_HERSHEY_DUPLEX
        r_font_color = (0, 255, 0) if not self.threshold_r_supression_mode else (255, 0 ,0)
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(self.img_rgb, 'R-channel:       {:.3f} ({:s})'.format(r_sum, thres), (1000, 160), font, font_scale, r_font_color, font_thickness)

        # threshold x-gradients
        r_gradients_x = self.cip.gradient(r_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='x')
        b_gradients_x = self.cip.gradient(b_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='x')
        r_gradients_y = self.cip.gradient(r_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='y')
        b_gradients_y = self.cip.gradient(b_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='y')
        img_binary_r_gradients_x = self.cip.abs_gradient_threshold(r_gradients_x, threshold=self.threshold_r_gradient_x)
        img_binary_b_gradients_x = self.cip.abs_gradient_threshold(b_gradients_x, threshold=self.threshold_b_gradient_x)

        # threshold magnitudes
        img_binary_r_magnitude = self.cip.magnitude_threshold(r_gradients_x, r_gradients_y, threshold=self.threshold_r_magnitude)
        img_binary_b_magnitude = self.cip.magnitude_threshold(b_gradients_x, b_gradients_y, threshold=self.threshold_b_magnitude)

        # debug images (colored RSB bindary images)
        self.img_binary_rlsb_3c = np.dstack((img_binary_r, img_binary_ls, img_binary_b))
        self.img_binary_rlsb_3c[self.img_binary_rlsb_3c == 1] = 255
        self.img_binary_x_grad_rb_3c = np.dstack((img_binary_r_gradients_x, np.zeros_like(img_binary_r), img_binary_b_gradients_x))
        self.img_binary_x_grad_rb_3c[self.img_binary_x_grad_rb_3c == 1] = 255
        self.img_binary_mag_rb_3c = np.dstack((img_binary_r_magnitude, np.zeros_like(img_binary_r), img_binary_b_magnitude))
        self.img_binary_mag_rb_3c[self.img_binary_mag_rb_3c == 1] = 255

        # combine binary images
        img_binary_combined = np.zeros_like(img_binary_r)
        img_binary_combined[(img_binary_r == 1) | (img_binary_b == 1) | (img_binary_ls == 1) |
                            (((img_binary_r_gradients_x == 1) | (img_binary_b_gradients_x == 1)) &
                              ((img_binary_r_magnitude == 1) | (img_binary_b_magnitude == 1)))] = 1

        if self.debug_preprocessing:
            self.cip.show_image(img_binary_combined, title='Pre-processing Result', cmap='gray')

        return img_binary_combined

    def fit_line(self, x, y, degree=2, coordinate_space='image'):
        """ Fits the line by a polynomial fit.

        Example second order polynomial:

            f(y) = A * y^2 + B * y + C

        :param x:                 x-values (array of points) in image space.
        :param y:                 y-values (array of points) in image space.
        :param degree:            Degree of polynomial. Default = second order polynomial
        :param coordinate_space:  'image' or 'world' coordinate space.

        :return: Returns the fitted polynomial coefficients (highest order first). Returns None in case of any error.
        """

        if coordinate_space == 'image':
            return np.polyfit(y, x, degree)
        elif coordinate_space == 'world':
            return np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, degree)
        else:
            return None

    def calc_curve_radius(self, x, y, yr, degree=2,):
        """ Calculated the curve radius in world space)

        :param x:         x-values (array of points) in image space.
        :param y:         y-values (array of points) in image space.
        :param yr:        y-position at which the curve radius shall be calculated in image space.
        :param degree:    Degree of polynomial. Default = second order polynomial

        :return: Returns the curve radius in meters (world space)
        """

        line_fit_w = self.fit_line(x, y, degree=2, coordinate_space='world')

        return ((1 + (2 * line_fit_w[0] * yr * self.ym_per_pix + line_fit_w[1])**2)**1.5) / np.absolute(2 * line_fit_w[0])

    def find_lines_with_sliding_window_histogram(self, nb_windows=9, margin=100, min_pixels=50):
        """ Finds and associates pixels for the left and right ego lane line based on a histogram search in the
        warped binary image.

        :param nb_windows: Number of sliding windows. Default = 9 windows
        :param margin:     Margin in pixels around the previous fitted polynomial. Default = +/- 100 pixels.
        :param min_pixels: Minimum number of pixels found to recenter window. Default = 50 pixels

        :return: Returns the left and right polynomial line fits and the x/y pixel coordinates.
                 left_line_fit, right_line_fit, line_fit_y, line_fit_left_x, line_fit_right_x
        """

        # take a histogram of the bottom half of the image
        histogram = np.sum(self.img_binary_warped[int(self.img_binary_warped.shape[0]/2):, :], axis=0)

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # set height of windows
        window_height = np.int(self.img_binary_warped.shape[0]/nb_windows)

        # identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img_binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # create empty lists to receive left and right lane pixel indices
        left_idx = []
        right_idx = []

        # step through the windows one by one
        for window in range(nb_windows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = self.img_binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.img_binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # draw the windows on the visualization image
            cv2.rectangle(self.img_line_search, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.img_line_search, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # identify the nonzero pixels in x and y within the window
            good_left_idx = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) &
                             (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_idx = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) &
                              (nonzero_x < win_xright_high)).nonzero()[0]

            # append these indices to the lists
            left_idx.append(good_left_idx)
            right_idx.append(good_right_idx)

            # if you found > minpix pixels, recenter next window on their mean position
            if len(good_left_idx) > min_pixels:
                leftx_current = np.int(np.mean(nonzero_x[good_left_idx]))

            if len(good_right_idx) > min_pixels:
                rightx_current = np.int(np.mean(nonzero_x[good_right_idx]))

        # concatenate the arrays of indices
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)

        # extract left and right line pixel positions
        left_x = nonzero_x[left_idx]
        left_y = nonzero_y[left_idx]
        right_x = nonzero_x[right_idx]
        right_y = nonzero_y[right_idx]

        # fit a second order polynomial to each
        left_line_fit = self.fit_line(left_x, left_y, degree=2, coordinate_space='image')
        right_line_fit = self.fit_line(right_x, right_y, degree=2, coordinate_space='image')

        # generate x and y values for plotting
        # f(y) = A * y^2 + B * y + C
        line_fit_y = np.linspace(0, self.img_binary_warped.shape[0] - 1, self.img_binary_warped.shape[0])
        line_fit_left_x = left_line_fit[0] * line_fit_y**2 + left_line_fit[1] * line_fit_y + left_line_fit[2]
        line_fit_right_x = right_line_fit[0] * line_fit_y**2 + right_line_fit[1] * line_fit_y + right_line_fit[2]

        # color lane marker pixels used by the polynomial fit (left = red, right = blue)
        self.img_line_search[nonzero_y[left_idx], nonzero_x[left_idx]] = [255, 0, 0]
        self.img_line_search[nonzero_y[right_idx], nonzero_x[right_idx]] = [0, 0, 255]

        return left_line_fit, right_line_fit, line_fit_y, line_fit_left_x, line_fit_right_x

    def find_lines_around_previous_lines(self, margin=100):
        """ Finds and associates pixels for the left and right ego lane line within an area around the detected line
        from the previous frame in the warped binary image.

        :param margin:  Margin in pixels around the previous fitted polynomial. Default = +/- 100 pixels.

        :return: Returns the left and right polynomial line fits and the x/y pixel coordinates.
                 left_line_fit, right_line_fit, line_fit_y, line_fit_left_x, line_fit_right_x
        """

        # search around the previous polynomial fit (+/- margin) instead of a full histogram based search
        nonzero = self.img_binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_idx = ((nonzero_x > (self.prev_left_line_fit[0] * (nonzero_y ** 2) + self.prev_left_line_fit[1] * nonzero_y + self.prev_left_line_fit[2] - margin)) &
                    (nonzero_x < (self.prev_left_line_fit[0] * (nonzero_y ** 2) + self.prev_left_line_fit[1] * nonzero_y + self.prev_left_line_fit[2] + margin)))
        right_idx = ((nonzero_x > (self.prev_right_line_fit[0] * (nonzero_y ** 2) + self.prev_right_line_fit[1] * nonzero_y + self.prev_right_line_fit[2] - margin)) &
                     (nonzero_x < (self.prev_right_line_fit[0] * (nonzero_y ** 2) + self.prev_right_line_fit[1] * nonzero_y + self.prev_right_line_fit[2] + margin)))

        # extract left and right line pixel positions in search area
        left_x = nonzero_x[left_idx]
        left_y = nonzero_y[left_idx]
        right_x = nonzero_x[right_idx]
        right_y = nonzero_y[right_idx]

        # fit a second order polynomial
        left_line_fit = self.fit_line(left_x, left_y, degree=2, coordinate_space='image')
        right_line_fit = self.fit_line(right_x, right_y, degree=2, coordinate_space='image')

        # generate x and y values for plotting
        # f(y) = A * y^2 + B * y + C
        line_fit_y = np.linspace(0, self.img_binary_warped.shape[0] - 1, self.img_binary_warped.shape[0])
        line_fit_left_x = left_line_fit[0] * line_fit_y**2 + left_line_fit[1] * line_fit_y + left_line_fit[2]
        line_fit_right_x = right_line_fit[0] * line_fit_y**2 + right_line_fit[1] * line_fit_y + right_line_fit[2]

        # color lane marker pixels used by the polynomial fit (left = red, right = blue)
        self.img_line_search[nonzero_y[left_idx], nonzero_x[left_idx]] = [255, 0, 0]
        self.img_line_search[nonzero_y[right_idx], nonzero_x[right_idx]] = [0, 0, 255]

        return left_line_fit, right_line_fit, line_fit_y, line_fit_left_x, line_fit_right_x

    def detect_ego_lane(self, img_rgb):
        """ Detects the ego lane.

        :param img_rgb:  Input RGB image (original image)

        :return: tbd.
        """

        #  pre-process image and warp to birds-eye-view
        self.img_rgb = img_rgb
        self.img_binary = self.preprocess_image(img_rgb)
        self.img_binary_warped = self.cip.warp(self.img_binary, self.warp_src_pts, self.warp_dst_pts,
                                               (self.warp_img_width, self.warp_img_height))

        # create an overlay image to draw detected ego lane lines and visualize debug information
        self.img_line_search = np.dstack((self.img_binary_warped, self.img_binary_warped, self.img_binary_warped)) * 255

        # detect ego lane lines
        margin = 100

        if self.mode == Mode.NO_DETECTION or self.prediction_count >= self.max_prediction_cycles:
            # no ego lane boundaries detected, search initial ones by sliding window histogram search
            left_line_fit, right_line_fit, line_fit_y, \
            line_fit_left_x, line_fit_right_x = self.find_lines_with_sliding_window_histogram(nb_windows=9,
                                                                                              margin=100,
                                                                                              min_pixels=50)
            self.mode = Mode.NEW_LINE
            self.prediction_count = 0
        else:
            # already detected ego lane boundaries in previous frame, search around previous polynomials
            left_line_fit, right_line_fit, line_fit_y, \
            line_fit_left_x, line_fit_right_x = self.find_lines_around_previous_lines(margin=margin)

            # TODO: average line fits (alphe = weight of previous line fit)
            #alpha = 0.5
            #left_line_fit = left_line_fit * (1 - alpha) + alpha * self.prev_left_line_fit
            #right_line_fit = right_line_fit * (1 - alpha) + alpha * self.prev_right_line_fitq

            self.mode = Mode.DETECTED

        # calculate curve radius in world space
        left_curve_radius = self.calc_curve_radius(line_fit_left_x, line_fit_y, np.max(line_fit_y), degree=2)
        right_curve_radius = self.calc_curve_radius(line_fit_right_x, line_fit_y, np.max(line_fit_y), degree=2)
        avg_curve_radius = (left_curve_radius + right_curve_radius) / 2.

        # calculate lane width, center offset and distance to left/right ego lane boundary
        # world coordinate system: x forwards, y left
        lane_width = (line_fit_right_x[-1] - line_fit_left_x[-1]) * self.xm_per_pix
        lane_center = (line_fit_right_x[-1] + line_fit_left_x[-1]) / 2.
        d_offset_center = (self.cam_pos_x - lane_center) * self.xm_per_pix
        d_left = (self.cam_pos_x - line_fit_left_x[-1]) * self.xm_per_pix
        d_right = (self.cam_pos_x - line_fit_right_x[-1]) * self.xm_per_pix

        # -----------------------------------------------------------------------
        # sanity checks

        sanity_check_lat_jump_failed = False

        if self.mode == Mode.DETECTED or self.mode == Mode.PREDICTED:
            # sanity check for lateral jumps
            d_jump = 0.25                                        # max lateral jump [m]
            delta_d_left = abs(abs(d_left) - abs(self.prev_d_left))
            delta_d_right = abs(abs(d_right) - abs(self.prev_d_right))

            if delta_d_left > d_jump or delta_d_right > d_jump:
                sanity_check_lat_jump_failed = True
                self.prediction_count += 1

            # use previous data if at least one sanitiy checks failes
            if sanity_check_lat_jump_failed:
                self.mode = Mode.PREDICTED
                line_fit_y = self.prev_line_fit_y
                line_fit_left_x = self.prev_line_fit_left_x
                line_fit_right_x = self.prev_line_fit_right_x
                left_line_fit = self.prev_left_line_fit
                right_line_fit = self.prev_right_line_fit
                left_curve_radius = self.prev_left_curve_radius
                right_curve_radius = self.prev_right_curve_radius
                avg_curve_radius = self.prev_avg_curve_radius
                d_left = self.prev_d_left
                d_right = self.prev_d_right
            else:
                self.prediction_count = 0

        # -----------------------------------------------------------------------
        # show textual information

        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 255)
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(self.img_rgb, 'left radius:    {:.2f} m'.format(left_curve_radius), (1000, 25), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'right radius:   {:.2f} m'.format(right_curve_radius), (1000, 40), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'delta radius:   {:.2f} m'.format(left_curve_radius - right_curve_radius), (1000, 55), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'avg L/R radius: {:.2f} m'.format(avg_curve_radius), (1000, 70), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'lane width:     {:.2f} m'.format(lane_width), (1000, 85), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'offset center:  {:.2f} m'.format(d_offset_center), (1000, 100), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'dist left:      {:.2f} m'.format(d_left), (1000, 115), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'dist right:     {:.2f} m'.format(d_right), (1000, 130), font, font_scale, font_color, font_thickness)

        if self.mode == Mode.NO_DETECTION:
            font_color = (255, 0, 0)
        elif self.mode == Mode.DETECTED:
            font_color = (0, 255, 0)
        elif self.mode == Mode.NEW_LINE:
            font_color = (255, 255, 0)
        elif self.mode == Mode.PREDICTED:
            font_color = (255, 255, 0)

        cv2.putText(self.img_rgb, 'Mode:            {:s}'.format(self.mode.name), (1000, 145), font, font_scale, font_color, font_thickness)

        if self.mode == Mode.DETECTED or self.mode == Mode.PREDICTED:
            font_color = (255, 0, 0) if sanity_check_lat_jump_failed else (0, 255, 0)
            cv2.putText(self.img_rgb, 'Sanity lat jump: {:.2f} {:.2f}'.format(delta_d_left, delta_d_right), (1000, 175), font, font_scale, font_color, font_thickness)

        # -----------------------------------------------------------------------
        # prepare overlay and debug plots

        img_overlay = np.zeros_like(self.img_line_search)

        # draw the search area onto the overlay image
        if not self.mode == Mode.NEW_LINE:
            # generate a polygon to illustrate the search window area
            # and recast the x and y points into usable format for cv2.fillPoly()
            left_line_window_1 = np.array([np.transpose(np.vstack([line_fit_left_x - margin, line_fit_y]))])
            left_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([line_fit_left_x + margin, line_fit_y])))])
            left_line_pts = np.hstack((left_line_window_1, left_line_window_2))

            right_line_window_1 = np.array([np.transpose(np.vstack([line_fit_right_x - margin, line_fit_y]))])
            right_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([line_fit_right_x + margin, line_fit_y])))])
            right_line_pts = np.hstack((right_line_window_1, right_line_window_2))

            img_search_area = np.zeros_like(self.img_line_search)
            cv2.fillPoly(img_search_area, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(img_search_area, np.int_([right_line_pts]), (0, 255, 0))

            self.img_line_search = cv2.addWeighted(self.img_line_search, 1, img_search_area, 0.3, 0)

            if self.overlay_ego_lane_search_area_enabled:
                img_overlay = cv2.addWeighted(img_overlay, 1, img_search_area, 0.3, 0)

        #  draw line fits onto overlay image
        left_line_pts = np.array([np.transpose(np.vstack([line_fit_left_x, line_fit_y]))])
        right_line_pts = np.array([np.transpose(np.vstack([line_fit_right_x, line_fit_y]))])
        cv2.polylines(self.img_line_search, np.int_([left_line_pts]), False, (0, 255, 0), 5)
        cv2.polylines(self.img_line_search, np.int_([right_line_pts]), False, (0, 255, 0), 5)

        if self.overlay_ego_lane_fits_enabled:
            cv2.polylines(img_overlay, np.int_([left_line_pts]), False, (0, 255, 0), 5)
            cv2.polylines(img_overlay, np.int_([right_line_pts]), False, (0, 255, 0), 5)

        # draw overlays onto original RGB image
        if self.mode == Mode.NEW_LINE or self.mode == Mode.DETECTED or self.mode == Mode.PREDICTED:
            # draw detected ego lane area
            if self.overlay_ego_lane_enabled:
                warped_zero = np.zeros_like(self.img_binary_warped).astype(np.uint8)
                color_warped = np.dstack((warped_zero, warped_zero, warped_zero))

                # recast the x and y points into usable format for cv2.fillPoly()
                pts_left = np.array([np.transpose(np.vstack([line_fit_left_x, line_fit_y]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([line_fit_right_x, line_fit_y])))])
                pts = np.hstack((pts_left, pts_right))

                # draw the ego lane onto the warped blank image
                cv2.fillPoly(color_warped, np.int_([pts]), (0, 255, 0))
                img_overlay = cv2.addWeighted(img_overlay, 1, color_warped, 0.3, 0)

            img_overlay_unwarped = self.cip.warp(img_overlay, self.warp_dst_pts, self.warp_src_pts)
            self.img_rgb = cv2.addWeighted(self.img_rgb, 1, img_overlay_unwarped, 0.9, 0)

        self.update_debug_plots()

        # store results for next cycle
        self.prev_line_fit_y = line_fit_y
        self.prev_line_fit_left_x = line_fit_left_x
        self.prev_line_fit_right_x = line_fit_right_x
        self.prev_left_line_fit = left_line_fit
        self.prev_right_line_fit = right_line_fit
        self.prev_left_curve_radius = left_curve_radius
        self.prev_right_curve_radius = right_curve_radius
        self.prev_avg_curve_radius = avg_curve_radius
        self.prev_d_left = d_left
        self.prev_d_right = d_right

def keyboard_thread():
    """ Keyboard input thread. """

    global running, start, end, idx, paused, step_one_frame, images

    while running:
        key = click.getchar()

        if key == 'q':
            running = False
            plt.close()
            print('Quit lane detection.')
        elif key == 'r':
            idx = start
            print('Restart lane detection ({:d}, {:d})'.format(start, end))
        elif key == '0':
            start = 0
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '1':
            start = 500
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '2':
            start = 970
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '3':
            start = 1010
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == 's':
            filename = 'frame_{:04d}.jpg'.format(idx)
            print('Safe image {:s}...'.format(filename), end='', flush=True)
            cv2.imwrite(filename, cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))
            print('done')
        elif key == 'p':
            if not paused:
                paused = not paused
                print('Paused at frame {:d}'.format(idx))
            else:
                paused = not paused
                print('Continued at frame {:d}'.format(idx))
        elif key == 'n':
            step_one_frame = True
            print('Step one frame {:d}'.format(idx))
        elif key == 'i':
            print('Frame index: {:d}'.format(idx))

if __name__ == '__main__':
    print('Advanced Lane Lines')

    ld = LaneDetection('calib.dat')
    ld.debug_preprocessing = False

    click_enabled = True
    running = True

    print('Extracting images from video file...', end='', flush=True)
    images = CoreImageProcessing.load_video('project_video.mp4')
    #images = CoreImageProcessing.load_video('challenge_video.mp4')
    #images = CoreImageProcessing.load_video('harder_challenge_video.mp4')
    print('done')

    start = 0
    end = len(images) - 1
    idx = start
    paused = False
    step_one_frame = False

    if click_enabled:
        thread = threading.Thread(target=keyboard_thread)
        thread.start()
        print('Started keyboard thread')

    while running:
        if idx > end:
            break

        if not paused or step_one_frame:
            ld.detect_ego_lane(images[idx])
            idx += 1
            step_one_frame = False

        plt.pause(0.0000001)

    if click_enabled:
        thread.join()

    plt.show()
