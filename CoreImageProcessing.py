import numpy as np
import cv2
import sys
import matplotlib
#matplotlib.use('macosx', force=True)  # does not supports all features on macos environments
matplotlib.use('TKAgg', force=True)   # slow but stable on macosx environments
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from Calibration import *

class CoreImageProcessing:
    """ Provides core image processing methods likes gradient calculation, gradient, direction, magnitude
    thresholds, etc. """

    def __init__(self):
        """ Initialization method. """

    @staticmethod
    def show_image(image, title='', cmap=None, show=False):
        """ Show a single image in a matplotlib figure.

        :param image:  Image to be shown.
        :param title:  Image title.
        :param cmap:   Colormap (most relevant: 'gray', 'jet', 'hsv')
                       For supported colormaps see: https://matplotlib.org/examples/color/colormaps_reference.html
        :param show:   If true, the image will be shown immediately. Otherwise `plt.show()` shall be called at a later
                       stage.
        """

        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

        if show:
            plt.show()

    @staticmethod
    def show_images(figsize, rows, images, titles=[], cmaps=[], fig_title='', show=False):
        """ Show a single image in a matplotlib figure.

        :param figsize:   Size of the image in inch (width, height).
        :param rows:      Number of rows.
        :param images:    1D-Array of images to be shown.
        :param titles:    1D-Array of image titles.
        :param cmaps:     1D-Array of colormaps (most relevant: 'gray', 'jet', 'hsv'). Use '' to apply default cmap.
                          For supported colormaps see: https://matplotlib.org/examples/color/colormaps_reference.html
        :param fig_title: Figure title.
        :param show:      If true, the image will be shown immediately. Otherwise `plt.show()` shall be called at a
                          later stage.
        """

        nb_images = len(images)
        nb_images_per_row = int(nb_images / rows)

        fig, axarr = plt.subplots(rows, nb_images_per_row, figsize=figsize)
        fig.tight_layout()

        if fig_title != '':
            fig.suptitle(fig_title)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.01, hspace=0.15)

        if rows == 1 or nb_images_per_row == 1:
            # plot single row
            for i, ax in enumerate(axarr):
                if cmaps[i] == '':
                    ax.imshow(images[i])
                else:
                    ax.imshow(images[i], cmap=cmaps[i])

                ax.set_title(titles[i])
                ax.axis('off')
        else:
            # plot multiple rows
            idx = 0
            for r in range(rows):
                for c in range(nb_images_per_row):
                    if cmaps[idx] == '':
                        axarr[r][c].imshow(images[idx])
                    else:
                        axarr[r][c].imshow(images[idx], cmap=cmaps[idx])

                    axarr[r][c].set_title(titles[idx])
                    axarr[r][c].axis('off')
                    idx += 1

        if show:
            plt.show()

    def threshold_image_channel(self, image_channel, threshold=(0, 255)):
        """ Thresholds a single image channel.

        :param image_channel: Single image channel (e.g. R, G, B or H, L, S channel)
        :param threshold:    Min/max color thresholds [0..255].

        :return: Returns a thresholded binary image.
        """

        binary = np.zeros_like(image_channel, dtype=np.uint8)
        binary[(image_channel >= threshold[0]) & (image_channel <= threshold[1])] = 1

        return binary

    def gradient(self, image, sobel_kernel=3, orientation='x'):
        """ Calculates the gradient of the image channel in x-, y- or in x- and y-direction.

        :param image:        Single channel image.
        :param sobel_kernel: Sobel kernel size. Min 3.
        :param orientation:  Gradient orientation ('x' = x-gradient, 'y' = y-gradient)

        :return: Returns the gradient or None in case of an error.
        """

        if orientation == 'x':
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orientation == 'y' or orientation == 'xy':
            return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            print('ERROR: Not supported gradient orientation (x or y supported only).', file=sys.stderr)
            return None

    def norm_abs_gradient(self, gradient):
        """ Calculates the normalized absolute directional gradients.

        :param gradient:     x- or y-gradients

        :return: Returns normalized [0..255] absolute gradient.
        """

        abs_gradient = np.absolute(gradient)
        return np.uint8(255 * abs_gradient / np.max(abs_gradient))

    def abs_gradient_threshold(self, gradient, threshold=(0, 255)):
        """ Calculates the absolute directional gradients and applies a threshold.

        :param gradient:     x- or y-gradients
        :param orientation:  Gradient orientation used for debug plots only.
                             ('' = no title, 'x' = x-gradient title, 'y' = y-gradient title)
        :param threshold:    Min/max thresholds of gradient [0..255].

        :return: Returns a thresholded gradient binary image.
        """

        abs_gradient = self.norm_abs_gradient(gradient)
        binary = np.zeros_like(abs_gradient)
        binary[(abs_gradient >= threshold[0]) & (abs_gradient <= threshold[1])] = 1

        return binary

    def norm_magnitude(self, gradient_x, gradient_y):
        """ Calculates the normalized magnitude of the x- and y-gradients.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients

        :return: Returns a normalized [0..255] magnitude.
        """

        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.uint8(255 * magnitude / np.max(magnitude))

    def magnitude_threshold(self, gradient_x, gradient_y, threshold=(0, 255)):
        """ Calculates the magnitude of the x- and y-gradients and applies a threshold.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients
        :param threshold:   Min/max thresholds of magnitude [0..255].

        :return: Returns a thresholded magnitude binary image.
        """

        magnitude = self.norm_magnitude(gradient_x, gradient_y)
        binary = np.zeros_like(magnitude)
        binary[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1

        return binary

    def direction(self, gradient_x, gradient_y):
        """ Calculates the direction of the absolute x- and y-gradients.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients

        :return: Returns the gradients' direction (angles).
        """

        return np.arctan2(np.absolute(gradient_y), np.absolute(gradient_x))

    def direction_threshold(self, gradient_x, gradient_y, threshold=(0, np.pi/2)):
        """ Calculates the direction of the absolute x- and y-gradients and applies a threshold.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients
        :param threshold:   Min/max thresholds of direction [0..PI/2].

        :return: Returns a thresholded direction binary image.
        """

        angles = self.direction(gradient_x, gradient_y)
        binary = np.zeros_like(angles)
        binary[(angles >= threshold[0]) & (angles <= threshold[1])] = 1

        return binary

    def warp(self, image, src_pts, dst_pts, dst_img_size=None):
        """ Warps an image from source points to the destination points.

        :param image:        Input image.
        :param src_pts:      Source points (at least 4 points required).
        :param dst_pts:      Destination points (at least 4 points required).
        :param dst_img_size: Size of destination image (width, height). If None, use source image size.

        :return: Returns the warp image.
        """

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if dst_img_size is None:
            dst_img_size = (image.shape[1], image.shape[0])

        return cv2.warpPerspective(image, M, dst_img_size, flags=cv2.INTER_LINEAR)

    @staticmethod
    def load_video(filename):
        """ Load video file and extract images.

        :param filename: mp4 filename

        :return: Returns extracted RGB images.
        """

        clip = VideoFileClip(filename)
        images = []

        for frame in clip.iter_frames():
            images.append(frame)

        return images
#
# Test environment for Core Image Processing class
#
def test_preprocessing_pipeline(img_rgb, plot_intermediate_results=False):
    """ Test optimization image pre-processing pipeline.

    :param img_rgb:                   Input RGB image.
    :param plot_intermediate_results: If true, all intermediate processing steps are plotted in separate figures.

    :return: Returns binary image representing all lane lines candidates.
    """

    # undistort image
    img_rgb = calib.undistort(img_rgb)

    # RGB channel thresholds
    threshold_r = (225, 255)                 # Threshold for R channel [0..255]
    threshold_g = (170, 255)                 # Threshold for G channel [0..255]
    threshold_b = (190, 255)                 # Threshold for B channel [0..255]

    # HLS channel thresholds
    threshold_h = (18, 25)                   # Threshold for H channel [0..179]
    threshold_l = (150, 255)                 # Threshold for L channel [0..255]
    threshold_s = (100, 255)                 # Threshold for S channel [0..255]

    sobel_kernel_rgb = 13                    # Sobel kernel size, odd number [3..31] for RGB image
    threshold_r_gradient_x = (25, 170)       # Threshold for R channel x-gradients [0..255]
    threshold_g_gradient_x = (30, 170)       # Threshold for G channel x-gradients [0..255]
    threshold_b_gradient_x = (20, 150)       # Threshold for B channel x-gradients [0..255]
    threshold_r_gradient_y = (25, 170)       # Threshold for R channel y-gradients [0..255]
    threshold_g_gradient_y = (30, 170)       # Threshold for G channel y-gradients [0..255]
    threshold_b_gradient_y = (30, 100)       # Threshold for B channel y-gradients [0..255]

    sobel_kernel_hls = 15                    # Sobel kernel size, odd number [3..31] for HLS image
    threshold_h_gradient_x = (5, 10)         # Threshold for H channel x-gradients [0..255]
    threshold_l_gradient_x = (35, 170)       # Threshold for S channel x-gradients [0..255]
    threshold_s_gradient_x = (30, 170)       # Threshold for L channel x-gradients [0..255]
    threshold_h_gradient_y = (5, 10)         # Threshold for H channel y-gradients [0..255]
    threshold_l_gradient_y = (80, 200)       # Threshold for S channel y-gradients [0..255]
    threshold_s_gradient_y = (40, 150)       # Threshold for L channel y-gradients [0..255]

    sobel_kernel_gray = 7                    # Sobel kernel size, odd number [3..31] for gray image
    threshold_gray_gradient_x = (25, 170)    # Threshold for gray image y-gradients [0..255]
    threshold_gray_gradient_y = (25, 170)    # Threshold for gray image y-gradients [0..255]

    threshold_r_magnitude = (25, 200)        # Threshold for R channel magnitude of x-/y-gradients [0..255]
    threshold_g_magnitude = (70, 200)        # Threshold for G channel magnitude of x-/y-gradients [0..255]
    threshold_b_magnitude = (35, 200)        # Threshold for B channel magnitude of x-/y-gradients [0..255]
    threshold_h_magnitude = (70, 200)        # Threshold for H channel magnitude of x-/y-gradients [0..255]
    threshold_l_magnitude = (70, 200)        # Threshold for L channel magnitude of x-/y-gradients [0..255]
    threshold_s_magnitude = (80, 200)        # Threshold for S channel magnitude of x-/y-gradients [0..255]
    threshold_gray_magnitude = (25, 200)     # Threshold for gray image magnitude of x-/y-gradients [0..255] --> 30

    threshold_r_direction = (0.85, 1.08)     # Threshold for R channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_g_direction = (0.85, 1.08)     # Threshold for G channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_b_direction = (0.85, 1.08)     # Threshold for B channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_h_direction = (0.60, 1.08)     # Threshold for H channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_l_direction = (0.85, 1.08)     # Threshold for L channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_s_direction = (0.85, 1.08)     # Threshold for S channel direction absolute x-/y-gradients [0..np.pi/2.]
    threshold_gray_direction = (0.8, 1.1)    # Threshold for gray image direction absolute x-/y-gradients [0..np.pi/2.]

    # -----------------------------------------------------------------------
    # RGB and HLS color thresholding

    # separate RGB and HLS channels
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    h_channel = img_hls[:, :, 0]
    l_channel = img_hls[:, :, 1]
    s_channel = img_hls[:, :, 2]

    # cip.show_images(figsize=(16, 6), rows=2,
    #                 images=[img_rgb, r_channel, g_channel, b_channel, img_hls, h_channel, l_channel, s_channel],
    #                 titles=['RGB Image', 'Red', 'Green', 'Blue', 'HLS Image', 'Hue', 'Lightness', 'Saturation'],
    #                 cmaps=['', 'gray', 'gray', 'gray', '', 'gray', 'gray', 'gray'])

    # threshold each channel
    img_binary_r = cip.threshold_image_channel(r_channel, threshold_r)
    img_binary_g = cip.threshold_image_channel(g_channel, threshold_g)
    img_binary_b = cip.threshold_image_channel(b_channel, threshold_b)
    img_binary_h = cip.threshold_image_channel(h_channel, threshold_h)
    img_binary_l = cip.threshold_image_channel(l_channel, threshold_l)
    img_binary_s = cip.threshold_image_channel(s_channel, threshold_s)

    if plot_intermediate_results:
        # cip.show_images(figsize=(16, 9), rows=3, fig_title='Image Channel Thresholding',
        #                 images=[r_channel, img_binary_r, h_channel, img_binary_h,
        #                         g_channel, img_binary_g, l_channel, img_binary_l,
        #                         b_channel, img_binary_b, s_channel, img_binary_s],
        #                 titles=['Red', 'Thresholded Red {:}'.format(threshold_r),
        #                         'Hue', 'Thresholded Hue {:}'.format(threshold_h),
        #                         'Green', 'Thresholded Green {:}'.format(threshold_g),
        #                         'Lightness', 'Thresholded Lightness {:}'.format(threshold_l),
        #                         'Blue', 'Thresholded Blue {:}'.format(threshold_b),
        #                         'Saturation', 'Thresholded Saturation {:}'.format(threshold_s)],
        #                 cmaps=['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray'])

        cip.show_images(figsize=(10, 9), rows=3, fig_title='Red, Blue, Saturation Thresholding',
                        images=[r_channel, img_binary_r,
                                b_channel, img_binary_b,
                                s_channel, img_binary_s,],
                        titles=['Red', 'Thresholded Red {:}'.format(threshold_r),
                                'Blue', 'Thresholded Blue {:}'.format(threshold_b),
                                'Saturation', 'Thresholded Saturation {:}'.format(threshold_s)],
                        cmaps=['gray', 'gray', 'gray', 'gray', 'gray', 'gray'])

    # -----------------------------------------------------------------------
    # RGB and HLS x/y gradient thresholding

    # calculate x- and y-gradients with sobel filter
    r_gradients_x = cip.gradient(r_channel, sobel_kernel=sobel_kernel_rgb, orientation='x')
    r_gradients_y = cip.gradient(r_channel, sobel_kernel=sobel_kernel_rgb, orientation='y')
    g_gradients_x = cip.gradient(g_channel, sobel_kernel=sobel_kernel_rgb, orientation='x')
    g_gradients_y = cip.gradient(g_channel, sobel_kernel=sobel_kernel_rgb, orientation='y')
    b_gradients_x = cip.gradient(b_channel, sobel_kernel=sobel_kernel_rgb, orientation='x')
    b_gradients_y = cip.gradient(b_channel, sobel_kernel=sobel_kernel_rgb, orientation='y')

    h_gradients_x = cip.gradient(h_channel, sobel_kernel=sobel_kernel_hls, orientation='x')
    h_gradients_y = cip.gradient(h_channel, sobel_kernel=sobel_kernel_hls, orientation='y')
    l_gradients_x = cip.gradient(l_channel, sobel_kernel=sobel_kernel_hls, orientation='x')
    l_gradients_y = cip.gradient(l_channel, sobel_kernel=sobel_kernel_hls, orientation='y')
    s_gradients_x = cip.gradient(s_channel, sobel_kernel=sobel_kernel_hls, orientation='x')
    s_gradients_y = cip.gradient(s_channel, sobel_kernel=sobel_kernel_hls, orientation='y')

    # apply gradient thresholds
    img_binary_r_gradient_x = cip.abs_gradient_threshold(r_gradients_x, threshold=threshold_r_gradient_x)
    img_binary_r_gradient_y = cip.abs_gradient_threshold(r_gradients_y, threshold=threshold_r_gradient_y)
    img_binary_g_gradient_x = cip.abs_gradient_threshold(g_gradients_x, threshold=threshold_g_gradient_x)
    img_binary_g_gradient_y = cip.abs_gradient_threshold(g_gradients_y, threshold=threshold_g_gradient_y)
    img_binary_b_gradient_x = cip.abs_gradient_threshold(b_gradients_x, threshold=threshold_b_gradient_x)
    img_binary_b_gradient_y = cip.abs_gradient_threshold(b_gradients_y, threshold=threshold_b_gradient_y)

    img_binary_h_gradient_x = cip.abs_gradient_threshold(h_gradients_x, threshold=threshold_h_gradient_x)
    img_binary_h_gradient_y = cip.abs_gradient_threshold(h_gradients_y, threshold=threshold_h_gradient_y)
    img_binary_l_gradient_x = cip.abs_gradient_threshold(l_gradients_x, threshold=threshold_l_gradient_x)
    img_binary_l_gradient_y = cip.abs_gradient_threshold(l_gradients_y, threshold=threshold_l_gradient_y)
    img_binary_s_gradient_x = cip.abs_gradient_threshold(s_gradients_x, threshold=threshold_s_gradient_x)
    img_binary_s_gradient_y = cip.abs_gradient_threshold(s_gradients_y, threshold=threshold_s_gradient_y)

    if plot_intermediate_results:
        # # plot RGB gradient thresholds
        # cip.show_images(figsize=(17, 7), rows=3, fig_title='RGB x/y-gradients',
        #                 images=[r_channel, cip.norm_abs_gradient(r_gradients_x), img_binary_r_gradient_x, cip.norm_abs_gradient(r_gradients_y), img_binary_r_gradient_y,
        #                         g_channel, cip.norm_abs_gradient(g_gradients_x), img_binary_g_gradient_x, cip.norm_abs_gradient(g_gradients_y), img_binary_g_gradient_y,
        #                         b_channel, cip.norm_abs_gradient(b_gradients_x), img_binary_b_gradient_x, cip.norm_abs_gradient(b_gradients_y), img_binary_b_gradient_y],
        #                 titles=['Red', 'x-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded x-gradient {:}'.format(threshold_r_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded y-gradient {:}'.format(threshold_r_gradient_y),
        #                         'Green', 'x-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded x-gradient {:}'.format(threshold_g_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded y-gradient {:}'.format(threshold_g_gradient_y),
        #                         'Blue', 'x-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded x-gradient {:}'.format( threshold_b_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded y-gradient {:}'.format(threshold_b_gradient_y)],
        #                 cmaps=['gray', 'jet', 'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray', 'jet', 'gray'])
        #
        # # plot HLS gradient thresholds
        # cip.show_images(figsize=(17, 7), rows=3, fig_title='HLS x/y-gradients',
        #                 images=[h_channel, cip.norm_abs_gradient(h_gradients_x), img_binary_h_gradient_x, cip.norm_abs_gradient(h_gradients_y), img_binary_h_gradient_y,
        #                         l_channel, cip.norm_abs_gradient(l_gradients_x), img_binary_l_gradient_x, cip.norm_abs_gradient(l_gradients_y), img_binary_l_gradient_y,
        #                         s_channel, cip.norm_abs_gradient(s_gradients_x), img_binary_s_gradient_x, cip.norm_abs_gradient(s_gradients_y), img_binary_s_gradient_y],
        #                 titles=['Hue', 'x-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded x-gradient {:}'.format(threshold_h_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded y-gradient {:}'.format(threshold_h_gradient_y),
        #                         'Lightness', 'x-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded x-gradient {:}'.format(threshold_l_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded y-gradient {:}'.format(threshold_l_gradient_y),
        #                         'Saturation', 'x-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded x-gradient {:}'.format(threshold_s_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded y-gradient {:}'.format(threshold_s_gradient_y)],
        #                 cmaps=['gray', 'jet', 'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray', 'jet', 'gray'])

        # plot RBS gradient thresholds
        cip.show_images(figsize=(17, 7), rows=3, fig_title='Red, Blue and Saturation x/y-gradients',
                        images=[r_channel, cip.norm_abs_gradient(r_gradients_x), img_binary_r_gradient_x, cip.norm_abs_gradient(r_gradients_y), img_binary_r_gradient_y,
                                b_channel, cip.norm_abs_gradient(g_gradients_x), img_binary_g_gradient_x, cip.norm_abs_gradient(b_gradients_y), img_binary_g_gradient_y,
                                s_channel, cip.norm_abs_gradient(s_gradients_x), img_binary_s_gradient_x, cip.norm_abs_gradient(s_gradients_y), img_binary_s_gradient_y],
                        titles=['Red', 'x-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded x-gradient {:}'.format(threshold_r_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded y-gradient {:}'.format(threshold_r_gradient_y),
                                'Blue', 'x-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded x-gradient {:}'.format(threshold_b_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_rgb), 'Thresholded y-gradient {:}'.format(threshold_b_gradient_y),
                                'Saturation', 'x-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded x-gradient {:}'.format(threshold_s_gradient_x), 'y-gradient kernel={:}'.format(sobel_kernel_hls), 'Thresholded y-gradient {:}'.format(threshold_s_gradient_y)],
                        cmaps=['gray', 'jet', 'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray', 'jet', 'gray'])

    # -----------------------------------------------------------------------
    # RGB and HLS magnitude thresholding

    r_magnitude = cip.norm_magnitude(r_gradients_x, r_gradients_y)
    g_magnitude = cip.norm_magnitude(g_gradients_x, g_gradients_y)
    b_magnitude = cip.norm_magnitude(b_gradients_x, b_gradients_y)
    h_magnitude = cip.norm_magnitude(h_gradients_x, h_gradients_y)
    l_magnitude = cip.norm_magnitude(l_gradients_x, l_gradients_y)
    s_magnitude = cip.norm_magnitude(s_gradients_x, s_gradients_y)

    img_binary_r_magnitude = cip.magnitude_threshold(r_gradients_x, r_gradients_y, threshold=threshold_r_magnitude)
    img_binary_g_magnitude = cip.magnitude_threshold(g_gradients_x, g_gradients_y, threshold=threshold_g_magnitude)
    img_binary_b_magnitude = cip.magnitude_threshold(b_gradients_x, b_gradients_y, threshold=threshold_b_magnitude)
    img_binary_h_magnitude = cip.magnitude_threshold(h_gradients_x, h_gradients_y, threshold=threshold_h_magnitude)
    img_binary_l_magnitude = cip.magnitude_threshold(l_gradients_x, l_gradients_y, threshold=threshold_l_magnitude)
    img_binary_s_magnitude = cip.magnitude_threshold(s_gradients_x, s_gradients_y, threshold=threshold_s_magnitude)

    if plot_intermediate_results:
        # # plot RGB magnitude thresholds
        # cip.show_images(figsize=(15, 9), rows=3, fig_title='RGB Magnitudes',
        #                 images=[r_channel, r_magnitude, img_binary_r_magnitude,
        #                         g_channel, g_magnitude, img_binary_g_magnitude,
        #                         b_channel, b_magnitude, img_binary_b_magnitude],
        #                 titles=['Red', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_r_magnitude),
        #                         'Green', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_g_magnitude),
        #                         'Blue', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_b_magnitude)],
        #                 cmaps=['gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray'])
        #
        # # plot HLS magnitude thresholds
        # cip.show_images(figsize=(15, 9), rows=3, fig_title='HLS Magnitudes',
        #                 images=[h_channel, h_magnitude, img_binary_h_magnitude,
        #                         l_channel, l_magnitude, img_binary_l_magnitude,
        #                         s_channel, s_magnitude, img_binary_s_magnitude],
        #                 titles=['Hue', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_h_magnitude),
        #                         'Lightness', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_l_magnitude),
        #                         'Saturation', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_s_magnitude)],
        #                 cmaps=['gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray'])

        # plot RBS magnitude thresholds
        cip.show_images(figsize=(15, 9), rows=3, fig_title='Red, Blue and Saturation Magnitudes',
                        images=[r_channel, r_magnitude, img_binary_r_magnitude,
                                b_channel, b_magnitude, img_binary_b_magnitude,
                                s_channel, s_magnitude, img_binary_s_magnitude],
                        titles=['Red', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_r_magnitude),
                                'Blue', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_b_magnitude),
                                'Saturation', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_s_magnitude)],
                        cmaps=['gray', 'jet', 'gray',
                               'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray'])

    # -----------------------------------------------------------------------
    # RGB and HLS direction thresholding

    r_direction = cip.direction(r_gradients_x, r_gradients_y)
    g_direction = cip.direction(g_gradients_x, g_gradients_y)
    b_direction = cip.direction(b_gradients_x, b_gradients_y)
    h_direction = cip.direction(h_gradients_x, h_gradients_y)
    l_direction = cip.direction(l_gradients_x, l_gradients_y)
    s_direction = cip.direction(s_gradients_x, s_gradients_y)

    img_binary_r_direction = cip.direction_threshold(r_gradients_x, r_gradients_y, threshold=threshold_r_direction)
    img_binary_g_direction = cip.direction_threshold(g_gradients_x, g_gradients_y, threshold=threshold_g_direction)
    img_binary_b_direction = cip.direction_threshold(b_gradients_x, b_gradients_y, threshold=threshold_b_direction)
    img_binary_h_direction = cip.direction_threshold(h_gradients_x, h_gradients_y, threshold=threshold_h_direction)
    img_binary_l_direction = cip.direction_threshold(l_gradients_x, l_gradients_y, threshold=threshold_l_direction)
    img_binary_s_direction = cip.direction_threshold(s_gradients_x, s_gradients_y, threshold=threshold_s_direction)

    if plot_intermediate_results:
        # # plot RGB magnitude thresholds
        # cip.show_images(figsize=(15, 9), rows=3, fig_title='RGB Directions',
        #                 images=[r_channel, r_direction, img_binary_r_direction,
        #                         g_channel, g_direction, img_binary_g_direction,
        #                         b_channel, b_direction, img_binary_b_direction],
        #                 titles=['Red', 'Direction', 'Thresholded Direction {:}'.format(threshold_r_direction),
        #                         'Green', 'Direction', 'Thresholded Direction {:}'.format(threshold_g_direction),
        #                         'Blue', 'Direction', 'Thresholded Direction {:}'.format(threshold_b_direction)],
        #                 cmaps=['gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray'])
        #
        # # plot HLS magnitude thresholds
        # cip.show_images(figsize=(15, 9), rows=3, fig_title='HLS Direction',
        #                 images=[h_channel, h_direction, img_binary_h_direction,
        #                         l_channel, l_direction, img_binary_l_direction,
        #                         s_channel, s_direction, img_binary_s_direction],
        #                 titles=['Hue', 'Direction', 'Thresholded Direction {:}'.format(threshold_h_direction),
        #                         'Lightness', 'Direction', 'Thresholded Direction {:}'.format(threshold_l_direction),
        #                         'Saturation', 'Direction', 'Thresholded Direction {:}'.format(threshold_s_direction)],
        #                 cmaps=['gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray',
        #                        'gray', 'jet', 'gray'])

        # plot RBS magnitude thresholds
        cip.show_images(figsize=(15, 9), rows=3, fig_title='Red, Blue, Saturation Directions',
                        images=[r_channel, r_direction, img_binary_r_direction,
                                b_channel, b_direction, img_binary_b_direction,
                                s_channel, s_direction, img_binary_s_direction],
                        titles=['Red', 'Direction', 'Thresholded Direction {:}'.format(threshold_r_direction),
                                'Blue', 'Direction', 'Thresholded Direction {:}'.format(threshold_b_direction),
                                'Saturation', 'Direction', 'Thresholded Direction {:}'.format(threshold_s_direction)],
                        cmaps=['gray', 'jet', 'gray',
                               'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray'])

    # -----------------------------------------------------------------------
    # sobel x/y, direction and magnitude on gray image

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_gradient_x = cip.gradient(img_gray, sobel_kernel=sobel_kernel_gray, orientation='x')
    gray_gradient_y = cip.gradient(img_gray, sobel_kernel=sobel_kernel_gray, orientation='y')
    gray_magnitute = cip.norm_magnitude(gray_gradient_x, gray_gradient_y)
    gray_direction = cip.direction(gray_gradient_x, gray_gradient_y)

    gray_binary_gradient_x = cip.abs_gradient_threshold(gray_gradient_x, threshold=threshold_gray_gradient_x)
    gray_binary_gradient_y = cip.abs_gradient_threshold(gray_gradient_y, threshold=threshold_gray_gradient_y)
    gray_binary_magnitude = cip.magnitude_threshold(gray_gradient_x, gray_gradient_y, threshold=threshold_gray_magnitude)
    gray_binary_direction = cip.direction_threshold(gray_gradient_x, gray_gradient_y, threshold=threshold_gray_direction)

    if plot_intermediate_results:
        # plot gray filter thresholds
        cip.show_images(figsize=(10, 9), rows=4, fig_title='Gray Filter',
                        images=[img_gray, cip.norm_abs_gradient(gray_gradient_x), gray_binary_gradient_x,
                                img_gray, cip.norm_abs_gradient(gray_gradient_y), gray_binary_gradient_y,
                                img_gray, gray_magnitute, gray_binary_magnitude,
                                img_gray, gray_direction, gray_binary_direction],
                        titles=['Gray', 'x-gradient kernel={:}'.format(sobel_kernel_gray), 'Thresholded x-gradient {:}'.format(threshold_gray_gradient_x),
                                'Gray', 'y-gradient kernel={:}'.format(sobel_kernel_gray), 'Thresholded y-gradient {:}'.format(threshold_gray_gradient_y),
                                'Gray', 'Magnitude', 'Thresholded Magnitude {:}'.format(threshold_gray_magnitude),
                                'Gray', 'Direction', 'Thresholded Direction {:}'.format(threshold_gray_direction)],
                        cmaps=['gray', 'jet', 'gray',
                               'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray',
                               'gray', 'jet', 'gray'])

    # -----------------------------------------------------------------------
    # combine binary images

    img_white = np.zeros_like(img_binary_r)
    img_white[0] = 1

    # color combinations
    img_binary_rsb_3c = np.dstack((img_binary_r, img_binary_s, img_binary_b))
    img_binary_rsb_3c[img_binary_rsb_3c == 1] = 255

    img_binary_rbs = np.zeros_like(img_binary_r)
    img_binary_rbs[(img_binary_r == 1) | (img_binary_b == 1) | (img_binary_s == 1)] = 1
    img_binary_rb_and_s = np.zeros_like(img_binary_r)
    img_binary_rb_and_s[((img_binary_r == 1) | (img_binary_b == 1)) & (img_binary_s == 1)] = 1

    # x gradient combinations
    img_binary_x_grad_rsb_3c = np.dstack((img_binary_r_gradient_x, img_binary_s_gradient_x, img_binary_b_gradient_x))
    img_binary_x_grad_rsb_3c[img_binary_x_grad_rsb_3c == 1] = 255

    img_binary_x_grad_rbs = np.zeros_like(img_binary_r)
    img_binary_x_grad_rbs[(img_binary_r_gradient_x == 1) | (img_binary_b_gradient_x == 1) | (img_binary_s_gradient_x == 1)] = 1
    img_binary_x_grad_rb_and_s = np.zeros_like(img_binary_r)
    img_binary_x_grad_rb_and_s[((img_binary_r_gradient_x == 1) | (img_binary_b_gradient_x == 1)) & (img_binary_s_gradient_x == 1)] = 1

    img_binary_rb_x_grad_grayS = np.zeros_like(img_binary_r)
    img_binary_rb_x_grad_grayS[(img_binary_r == 1) | (img_binary_b == 1) | (gray_gradient_x == 1) | (img_binary_s_gradient_x == 1)] = 1

    img_binary_x_grad_rsb_3c = np.dstack((img_binary_r_gradient_x, img_binary_s_gradient_x, img_binary_b_gradient_x))
    img_binary_x_grad_rsb_3c[img_binary_x_grad_rsb_3c == 1] = 255
    img_binary_x_grad_rbs = np.zeros_like(img_binary_r)
    img_binary_x_grad_rbs[(img_binary_r_gradient_x == 1) | (img_binary_b_gradient_x == 1) | (img_binary_s_gradient_x == 1)] = 1

    img_binary_rbs_and_x_grad_rbs = np.zeros_like(img_binary_r)
    img_binary_rbs_and_x_grad_rbs[(img_binary_rbs == 1) & (img_binary_x_grad_rbs == 1)] = 1

    img_binary_rbs_gray_dir_and_gray_mag = np.zeros_like(img_binary_r)
    img_binary_rbs_gray_dir_and_gray_mag[((img_binary_rbs == 1) | (gray_direction == 1)) & (gray_magnitute == 1)] = 1

    if plot_intermediate_results:
        cip.show_images(figsize=(17, 9), rows=3, fig_title='Finally combined Binary Images',
                        images=[img_rgb, img_binary_rsb_3c, img_binary_rbs, img_binary_rb_and_s,
                                img_white, img_binary_x_grad_rsb_3c, img_binary_x_grad_rbs, img_binary_x_grad_rb_and_s,
                                img_white, img_binary_rbs_and_x_grad_rbs, img_binary_rb_x_grad_grayS, img_binary_rbs_gray_dir_and_gray_mag],
                        titles=['RGB Image', 'Binary RSB Image', 'Binary R|B|S Image', 'Binary (R|B)&S Image',
                                '', 'Binary x-grad RSB', 'Binary x-grad R|B|S', 'Binary x-grad (R|B)&S',
                                '', 'Binary R|B|S and x-grad R|B|S', 'Binary (R|B) & (x-grad gray|S)', '(R|B|S | gray dir) & gray mag'],
                        cmaps=['', '', 'gray', 'gray',
                               'gray', '', 'gray', 'gray',
                               'gray', 'gray', 'gray', 'gray'])

    return img_binary_rbs_and_x_grad_rbs


def test_warp(img_rgb):
    """ Test image warping.

    :param img_rgb: Input RGB image.
    """

    # undistort image
    img_rgb = calib.undistort(img_rgb)

    # point order = btm left --> btm right --> btm left --> btm right
    height = img_rgb.shape[0]
    src_pts = np.float32([[193, height], [1117, height], [689, 450], [592, 450]])
    dst_pts = np.float32([[300, height], [977, height], [977, 0], [300, 0]])

    img_warped = cip.warp(img_rgb, src_pts, dst_pts)
    cv2.polylines(img_rgb, np.int32([src_pts]), isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(img_warped, np.int32([dst_pts]), isClosed=True, color=(255, 0, 0), thickness=2)

    cip.show_images(figsize=(10, 4), rows=1,
                    images=[img_rgb, img_warped],
                    titles=['Original Image', 'Warped Image'],
                    cmaps=['', ''])
    plt.show()


if __name__ == '__main__':
    print('-----------------------------------------------------------------------------')
    print(' CIP - Core Image Processing Tests')
    print('-----------------------------------------------------------------------------')

    # load calibration data
    calib = Calibration('calib.dat')

    # configure core image processing
    cip = CoreImageProcessing()
    cip.debug_threshold_methods = True

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
        print('Load image file: {:s}'.format(f))
        img_rgb.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))

    # -----------------------------------------------------------------------
    # Test and optimize single images
    test_preprocessing_pipeline(img_rgb[0], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[1], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[2], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[6], plot_intermediate_results=True)
    plt.show()
    exit(0)

    # -----------------------------------------------------------------------
    # test warping
    # test_warp(img_rgb[0])
    # exit(0)

    # -----------------------------------------------------------------------
    # Pre-process all test images
    img_preprocessed = []
    titles = []
    cmaps = []

    for i, img in enumerate(img_rgb):
        print('Pre-process image {:s}'. format(img_files[i]))
        img_preprocessed.append(img)
        img_preprocessed.append(test_preprocessing_pipeline(img))
        titles.extend([img_files[i], '(R&S)&B Binary Image'])
        cmaps.extend(['', 'gray'])

    cip.show_images(figsize=(12, 9), rows=4, fig_title='Pre-processing Results',
                    images=img_preprocessed,
                    titles=titles,
                    cmaps=cmaps)

    plt.draw()
    plt.pause(1e-3)
    plt.show()
