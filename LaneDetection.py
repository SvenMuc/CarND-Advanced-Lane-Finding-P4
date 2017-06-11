import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('macosx', force=True)
#matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from Calibration import *
from CoreImageProcessing import *


class LaneDetection:
    """ Detects the ego lane (left and right line markings) from RGB images. """

    # enable debug for individual processing steps
    debug_preprocessing = False            # If true, plot pre-processing debug data

    calib = None                           # Camera calibration to undistort the images
    cip = None                             # Core image processing instance

    # -----------------------------------------------------------------------
    # Image pre-processing parameters

    # RGB channel thresholds
    threshold_r = (225, 255)               # Threshold for R channel [0..255]
    threshold_g = (170, 255)               # Threshold for G channel [0..255]
    threshold_b = (190, 255)               # Threshold for B channel [0..255]

    # HLS channel thresholds
    threshold_h = (18, 25)                 # Threshold for H channel [0..179]
    threshold_l = (150, 255)               # Threshold for L channel [0..255]
    threshold_s = (100, 255)               # Threshold for S channel [0..255]

    sobel_kernel_rgb = 13                  # Sobel kernel size, odd number [3..31] for RGB image
    threshold_r_gradient_x = (25, 170)     # Threshold for R channel x-gradients [0..255]
    threshold_b_gradient_x = (20, 150)     # Threshold for B channel x-gradients [0..255]

    sobel_kernel_hls = 15                  # Sobel kernel size, odd number [3..31] for HLS image
    threshold_s_gradient_x = (30, 170)     # Threshold for L channel x-gradients [0..255]

    # -----------------------------------------------------------------------
    # Find lane lines parameters

    # warping rectangles
    warp_img_height = 720
    warp_img_width = 1280
    warp_src_pts = np.float32([[193, warp_img_height], [1117, warp_img_height], [689, 450], [592, 450]])
    warp_dst_pts = np.float32([[300, warp_img_height], [977, warp_img_height], [977, 0], [300, 0]])

    detected = False                       # was a line detected in the previous frame?
    new_line = False                       # If true, the lane line is detected in the current frame

    prev_left_line_fit = None              # left polynomial line fit from previous frame
    prev_right_line_fit = None             # right polynomial line fit from previous frame
    prev_left_curve_radius = None          # left curve radius in meter from previous frame
    prev_right_curve_radius = None         # right curve radius in meter from previous frame

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

    def __init__(self, calibration_file):
        """ Initialization method.

        :param calibration_file:           Camera calibration file.
        """

        self.calib = Calibration(calibration_file)
        self.cip = CoreImageProcessing()

        self.init_debug_plots()

        # TODO: self.recent_xfitted = []                          # x values of the last n fits of the line
        # self.bestx = None                                 # average x values of fitted line over the last n iterations
        # self.best_fit = None                              # polynomial coefficients averaged over the last n iterations
        # self.current_fit = [np.array([False])]            # polynomial coefficients for the most recent fit
        # self.radius_of_curvature = None                   # radius of curvature of the line in some units
        # self.line_base_pos = None                         # distance in meters of vehicle center from the line
        # self.diffs = np.array([0, 0, 0], dtype='float')   # difference in fit coefficients between last and new fits
        # self.allx = None                                  # x values for detected line pixels
        # self.ally = None                                  # y values for detected line pixels

    def init_debug_plots(self):
        """ Initialize figure for debugging plots. """

        self.fig = plt.figure(figsize=(13, 7))
        self.fig.subplots_adjust(bottom=0.01, left=0.01, top=0.99, right=0.99, wspace=0.01, hspace=0.01)
        self.debug_plot.append(plt.imshow(self.img_rgb))
        plt.axis('off')
        # TODO: self.debug_plot.append(self.fig.add_axes([0.1, 0.8, 0.2, 0.2]))

    def update_debug_plots(self):
        """ Updates the debug plots with latest pipeline results. """

        img_rgb_overlay = self.img_rgb.copy()

        img = cv2.resize(cv2.cvtColor(self.img_binary, cv2.COLOR_GRAY2RGB), (320, 180))
        img_rgb_overlay[10:190, 10:330, :] = img * 255

        img = cv2.resize(cv2.cvtColor(self.img_binary_warped, cv2.COLOR_GRAY2RGB), (320, 180))
        img_rgb_overlay[10:190, 340:660, :] = img * 255

        img = cv2.resize(self.img_line_search, (320, 180))
        img_rgb_overlay[10:190, 670:990, :] = img

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
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # separate RGB and HLS channels
        r_channel = img_rgb[:, :, 0]
        b_channel = img_rgb[:, :, 2]

        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        s_channel = img_hls[:, :, 2]

        # threshold color channels
        img_binary_r = self.cip.threshold_image_channel(r_channel, self.threshold_r)
        img_binary_b = self.cip.threshold_image_channel(b_channel, self.threshold_b)
        img_binary_s = self.cip.threshold_image_channel(s_channel, self.threshold_s)

        # threshold x-gradients
        r_gradients_x = self.cip.gradient(r_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='x')
        b_gradients_x = self.cip.gradient(b_channel, sobel_kernel=self.sobel_kernel_rgb, orientation='x')
        s_gradients_x = self.cip.gradient(s_channel, sobel_kernel=self.sobel_kernel_hls, orientation='x')
        img_binary_r_gradients_x = self.cip.abs_gradient_threshold(r_gradients_x, threshold=self.threshold_r_gradient_x)
        img_binary_b_gradients_x = self.cip.abs_gradient_threshold(b_gradients_x, threshold=self.threshold_b_gradient_x)
        img_binary_s_gradients_x = self.cip.abs_gradient_threshold(s_gradients_x, threshold=self.threshold_s_gradient_x)

        # combine binary images
        img_binary_combined = np.zeros_like(img_binary_r)
        img_binary_combined[((img_binary_r == 1) | (img_binary_s == 1) | (img_binary_b == 1)) &
                            ((img_binary_r_gradients_x == 1) | (img_binary_b_gradients_x == 1) | (img_binary_s_gradients_x == 1))] = 1

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

    def fine_lines_around_previous_lines(self, margin=100):
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

        if not self.detected:
            # no ego lane boundaries detected, search initial ones by sliding window histogram search
            left_line_fit, right_line_fit, line_fit_y, \
            line_fit_left_x, line_fit_right_x = self.find_lines_with_sliding_window_histogram(nb_windows=9,
                                                                                              margin=100,
                                                                                              min_pixels=50)
            self.detected = True
            self.new_line = True
        else:
            # already detected ego lane boundaries in previous frame, search around previous polynomials
            left_line_fit, right_line_fit, line_fit_y, \
            line_fit_left_x, line_fit_right_x = self.fine_lines_around_previous_lines(margin=margin)

            self.new_line = False

        # calculate curve radius in world space
        left_curve_radius = self.calc_curve_radius(line_fit_left_x, line_fit_y, np.max(line_fit_y), degree=2)
        right_curve_radius = self.calc_curve_radius(line_fit_right_x, line_fit_y, np.max(line_fit_y), degree=2)

        # calculate lane width, center offset and distance to left/right ego lane boundary
        # world coordinate system: x forwards, y left
        lane_width = (line_fit_right_x[-1] - line_fit_left_x[-1]) * self.xm_per_pix
        lane_center = (line_fit_right_x[-1] + line_fit_left_x[-1]) / 2.
        d_offset_center = (self.cam_pos_x - lane_center) * self.xm_per_pix
        d_left = (self.cam_pos_x - line_fit_left_x[-1]) * self.xm_per_pix
        d_right = (self.cam_pos_x - line_fit_right_x[-1]) * self.xm_per_pix

        # -----------------------------------------------------------------------
        # show textual information

        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 255)
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(self.img_rgb, 'left radius:    {:7.2f} m'.format(left_curve_radius), (1000, 25), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'right radius:   {:7.2f} m'.format(right_curve_radius), (1000, 40), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'delta radius:   {:7.2f} m'.format(left_curve_radius - right_curve_radius), (1000, 55), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'avg L/R radius: {:7.2f} m'.format((left_curve_radius + right_curve_radius) / 2.), (1000, 70), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'lane width:     {:7.2f} m'.format(lane_width), (1000, 85), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'offset center:  {:7.2f} m'.format(d_offset_center), (1000, 100), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'dist left:      {:7.2f} m'.format(d_left), (1000, 115), font, font_scale, font_color, font_thickness)
        cv2.putText(self.img_rgb, 'dist right:     {:7.2f} m'.format(d_right), (1000, 130), font, font_scale, font_color, font_thickness)

        # -----------------------------------------------------------------------
        # prepare overlay and debug plots

        img_overlay = np.zeros_like(self.img_line_search)

        # draw the search area onto the overlay image
        if not self.new_line:
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
        if self.detected:
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
        self.prev_left_line_fit = left_line_fit
        self.prev_right_line_fit = right_line_fit
        self.prev_left_curve_radius = left_curve_radius
        self.prev_right_curve_radius = right_curve_radius

if __name__ == '__main__':
    print('Advanced Lane Lines')

    ld = LaneDetection('calib.dat')
    ld.debug_preprocessing = False
    ld.debug_fine_lane_lines = False

    use_video_file = True

    if not use_video_file:
        # use test images
        img_files = ['test_images/straight_lines1.jpg',
                                                   'test_images/straight_lines2.jpg',
                                                   'test_images/test1.jpg',
                                                   'test_images/test2.jpg',
                                                   'test_images/test3.jpg',
                                                   'test_images/test4.jpg',
                                                   'test_images/test5.jpg',
                                                   'test_images/test6.jpg']
        img_rgb = []

        for f in img_files:
            img_rgb.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))

        for img in img_rgb:
            ld.detect_ego_lane(img)
            plt.pause(0.5)

        plt.show()
    else:
        # use video file
        print('Extracting images from video file...', end='', flush=True)
        images = CoreImageProcessing.load_video('project_video.mp4')
        #images = CoreImageProcessing.load_video('challenge_video.mp4')
        #images = CoreImageProcessing.load_video('harder_challenge_video.mp4')
        print('done')

        for img in images:
            ld.detect_ego_lane(img)
            plt.pause(0.00001)

        plt.show()
