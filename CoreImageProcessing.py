import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('macosx', force=True)  # does not supports all features on macos environments
#matplotlib.use('TKAgg', force=True)   # slow but stable on macosx environments
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
            plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.01)

        if rows == 1 or nb_images_per_row == 1:
            # plot single row
            for i, ax in enumerate(axarr):
                if cmaps[i] == '':
                    ax.imshow(images[i])
                else:
                    ax.imshow(images[i], cmap=cmaps[i])

                ax.set_title(titles[i])
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

        angles = self.direction_threshold(gradient_x, gradient_y)
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

    sobel_kernel = 9                         # Sobel kernel size [3..21]

    threshold_r_gradient_x = (25, 170)       # Threshold for R channel x-gradients [0..255]
    threshold_g_gradient_x = (50, 150)       # Threshold for G channel x-gradients [0..255]
    threshold_b_gradient_x = (30, 170)       # Threshold for B channel x-gradients [0..255]
    threshold_r_gradient_y = (30, 170)       # Threshold for R channel y-gradients [0..255]
    threshold_g_gradient_y = (30, 170)       # Threshold for G channel y-gradients [0..255]
    threshold_b_gradient_y = (30, 170)       # Threshold for B channel y-gradients [0..255]

    threshold_h_gradient_x = (5, 10)         # Threshold for H channel x-gradients [0..255]
    threshold_l_gradient_x = (50, 170)       # Threshold for S channel x-gradients [0..255]
    threshold_s_gradient_x = (50, 170)       # Threshold for L channel x-gradients [0..255]
    threshold_h_gradient_y = (5, 10)         # Threshold for H channel y-gradients [0..255]
    threshold_l_gradient_y = (80, 200)       # Threshold for S channel y-gradients [0..255]
    threshold_s_gradient_y = (75, 220)       # Threshold for L channel y-gradients [0..255]

    threshold_magnitude = (70, 200)          # Threshold for magnitude of x-/y-gradients [0..255]
    threshold_direction = (0.85, 1.08)       # Threshold for direction absolute x-/y-gradients [0..np.pi/2.]

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
        cip.show_images(figsize=(16, 9), rows=3, fig_title='Image Channel Thresholding',
                        images=[r_channel, img_binary_r, h_channel, img_binary_h,
                                g_channel, img_binary_g, l_channel, img_binary_l,
                                b_channel, img_binary_b, s_channel, img_binary_s],
                        titles=['Red', 'Thresholded Red {:}'.format(threshold_r),
                                'Hue', 'Thresholded Hue {:}'.format(threshold_h),
                                'Green', 'Thresholded Green {:}'.format(threshold_g),
                                'Lightness', 'Thresholded Lightness  {:}'.format(threshold_l),
                                'Blue', 'Thresholded Blue {:}'.format(threshold_b),
                                'Saturation', 'Thresholded Saturation {:}'.format(threshold_s)],
                        cmaps=['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray'])

    # combine binary image channels (2 channels)
    # img_rb_binary_3c = np.dstack((img_binary_r, np.zeros_like(img_binary_r), img_binary_b))
    # img_rh_binary_3c = np.dstack((img_binary_r, np.zeros_like(img_binary_r), img_binary_h))
    # img_rs_binary_3c = np.dstack((img_binary_r, np.zeros_like(img_binary_r), img_binary_s))
    # img_bh_binary_3c = np.dstack((img_binary_b, np.zeros_like(img_binary_r), img_binary_h))
    # img_bs_binary_3c = np.dstack((img_binary_b, np.zeros_like(img_binary_r), img_binary_s))
    # img_hs_binary_3c = np.dstack((img_binary_h, np.zeros_like(img_binary_r), img_binary_s))
    #
    # img_rb_binary_3c[img_rb_binary_3c == 1] = 255
    # img_rh_binary_3c[img_rh_binary_3c == 1] = 255
    # img_rs_binary_3c[img_rs_binary_3c == 1] = 255
    # img_bs_binary_3c[img_bs_binary_3c == 1] = 255
    # img_bh_binary_3c[img_bh_binary_3c == 1] = 255
    # img_hs_binary_3c[img_hs_binary_3c == 1] = 255
    #
    # img_rb_binary = np.zeros_like(img_binary_r)
    # img_rb_binary[(img_binary_r == 1) | (img_binary_b == 1)] = 1
    # img_rh_binary = np.zeros_like(img_binary_r)
    # img_rh_binary[(img_binary_r == 1) | (img_binary_h == 1)] = 1
    # img_rs_binary = np.zeros_like(img_binary_r)
    # img_rs_binary[(img_binary_r == 1) | (img_binary_s == 1)] = 1
    # img_bs_binary = np.zeros_like(img_binary_r)
    # img_bs_binary[(img_binary_b == 1) | (img_binary_s == 1)] = 1
    # img_bh_binary = np.zeros_like(img_binary_r)
    # img_bh_binary[(img_binary_b == 1) | (img_binary_h == 1)] = 1
    # img_hs_binary = np.zeros_like(img_binary_r)
    # img_hs_binary[(img_binary_h == 1) | (img_binary_s == 1)] = 1

    # cip.show_images(figsize=(16, 5), rows=2, fig_title='Combined Binary Images',
    #                 images=[img_rb_binary_3c, img_rh_binary_3c, img_rs_binary_3c, img_bs_binary_3c, img_bh_binary_3c, img_hs_binary_3c,
    #                         img_rb_binary, img_rh_binary, img_rs_binary, img_bs_binary, img_bh_binary, img_hs_binary],
    #                 titles=['RB Binary Image', 'RH Binary Image', 'RS Binary Image', 'BS Binary Image', 'BH Binary Image', 'HS Binary Image',
    #                         'R|B Binary Image', 'R|H Binary Image', 'R|S Binary Image', 'B|S Binary Image', 'B|H Binary Image', 'H|S Binary Image'],
    #                 cmaps=['', '', '', '', '', '', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray'])

    # combine binary image channels (3 channels)
    img_rbs_binary_3c = np.dstack((img_binary_r, img_binary_b, img_binary_s))
    img_rbs_binary_3c[img_rbs_binary_3c == 1] = 255

    img_rbs_binary = np.zeros_like(img_binary_r)
    img_rbs_binary[(img_binary_r == 1) | (img_binary_b == 1) | (img_binary_s == 1)] = 1
    img_rb_and_s_binary = np.zeros_like(img_binary_r)
    img_rb_and_s_binary[((img_binary_r == 1) | (img_binary_b == 1)) & (img_binary_s == 1)] = 1
    img_r_and_s_or_b_binary = np.zeros_like(img_binary_r)
    img_r_and_s_or_b_binary[((img_binary_r == 1) & (img_binary_s == 1)) | (img_binary_b == 1)] = 1

    if plot_intermediate_results:
        cip.show_images(figsize=(17, 3), rows=1, fig_title='Finally combined Binary Images',
                        images=[img_rgb, img_rbs_binary_3c, img_rbs_binary, img_rb_and_s_binary, img_r_and_s_or_b_binary],
                        titles=['RGB Image', 'RBS Binary Image', 'R|B|S Binary Image', '(R|B)&S Binary Image', '(R&S)&B Binary Image'],
                        cmaps=['', '', 'gray', 'gray', 'gray'])

    return img_r_and_s_or_b_binary


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

    # Test and optimize single images
    # test_preprocessing_pipeline(img_rgb[0], plot_intermediate_results=True)
    # test_preprocessing_pipeline(img_rgb[1], plot_intermediate_results=True)
    # test_preprocessing_pipeline(img_rgb[2], plot_intermediate_results=True)
    test_preprocessing_pipeline(img_rgb[6], plot_intermediate_results=True)
    plt.show()
    exit(0)

    # TODO: test warping
    # test_warp(img_rgb[0])
    # exit(0)

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

    # # calculate x- and y-gradients with sobel filter
    # r_gradients_x = cip.gradient(r_channel, sobel_kernel=sobel_kernel, orientation='x')
    # r_gradients_y = cip.gradient(r_channel, sobel_kernel=sobel_kernel, orientation='y')
    # g_gradients_x = cip.gradient(g_channel, sobel_kernel=sobel_kernel, orientation='x')
    # g_gradients_y = cip.gradient(g_channel, sobel_kernel=sobel_kernel, orientation='y')
    # b_gradients_x = cip.gradient(b_channel, sobel_kernel=sobel_kernel, orientation='x')
    # b_gradients_y = cip.gradient(b_channel, sobel_kernel=sobel_kernel, orientation='y')
    #
    # h_gradients_x = cip.gradient(h_channel, sobel_kernel=sobel_kernel, orientation='x')
    # h_gradients_y = cip.gradient(h_channel, sobel_kernel=sobel_kernel, orientation='y')
    # l_gradients_x = cip.gradient(l_channel, sobel_kernel=sobel_kernel, orientation='x')
    # l_gradients_y = cip.gradient(l_channel, sobel_kernel=sobel_kernel, orientation='y')
    # s_gradients_x = cip.gradient(s_channel, sobel_kernel=sobel_kernel, orientation='x')
    # s_gradients_y = cip.gradient(s_channel, sobel_kernel=sobel_kernel, orientation='y')
    #
    # # apply gradient thresholds
    # img_binary_r_gradient_x = cip.abs_gradient_threshold(r_gradients_x, threshold=threshold_r_gradient_x)
    # img_binary_r_gradient_y = cip.abs_gradient_threshold(r_gradients_y, threshold=threshold_r_gradient_y)
    # img_binary_g_gradient_x = cip.abs_gradient_threshold(g_gradients_x, threshold=threshold_g_gradient_x)
    # img_binary_g_gradient_y = cip.abs_gradient_threshold(g_gradients_y, threshold=threshold_g_gradient_y)
    # img_binary_b_gradient_x = cip.abs_gradient_threshold(b_gradients_x, threshold=threshold_b_gradient_x)
    # img_binary_b_gradient_y = cip.abs_gradient_threshold(b_gradients_y, threshold=threshold_b_gradient_y)
    #
    # img_binary_h_gradient_x = cip.abs_gradient_threshold(h_gradients_x, threshold=threshold_h_gradient_x)
    # img_binary_h_gradient_y = cip.abs_gradient_threshold(h_gradients_y, threshold=threshold_h_gradient_y)
    # img_binary_l_gradient_x = cip.abs_gradient_threshold(l_gradients_x, threshold=threshold_l_gradient_x)
    # img_binary_l_gradient_y = cip.abs_gradient_threshold(l_gradients_y, threshold=threshold_l_gradient_y)
    # img_binary_s_gradient_x = cip.abs_gradient_threshold(s_gradients_x, threshold=threshold_s_gradient_x)
    # img_binary_s_gradient_y = cip.abs_gradient_threshold(s_gradients_y, threshold=threshold_s_gradient_y)

    # plot RGB gradient thresholds
    # cip.show_images(figsize=(16, 7), rows=2, fig_title='RGB x-gradients',
    #                 images=[cip.norm_abs_gradient(r_gradients_x), cip.norm_abs_gradient(g_gradients_x), cip.norm_abs_gradient(b_gradients_x),
    #                         img_binary_r_gradient_x, img_binary_g_gradient_x, img_binary_b_gradient_x],
    #                 titles=['Red x-gradient', 'Green x-gradient', 'Blue x-gradient',
    #                         'Thresholded Red x-gradient {:}'.format(threshold_r_gradient_x),
    #                         'Thresholded Green x-gradient {:}'.format(threshold_g_gradient_x),
    #                         'Thresholded Blue x-gradient {:}'.format(threshold_b_gradient_x)],
    #                 cmaps=['jet', 'jet', 'jet', 'gray', 'gray', 'gray'])
    #
    # cip.show_images(figsize=(16, 7), rows=2, fig_title='RGB y-gradients',
    #                 images=[cip.norm_abs_gradient(r_gradients_y), cip.norm_abs_gradient(g_gradients_y), cip.norm_abs_gradient(b_gradients_y),
    #                         img_binary_r_gradient_y, img_binary_g_gradient_y, img_binary_b_gradient_y],
    #                 titles=['Red y-gradient', 'Green y-gradient', 'Blue y-gradient',
    #                         'Thresholded Red y-gradient {:}'.format(threshold_r_gradient_y),
    #                         'Thresholded Green y-gradient {:}'.format(threshold_b_gradient_y),
    #                         'Thresholded Blue y-gradient {:}'.format(threshold_b_gradient_y)],
    #                 cmaps=['jet', 'jet', 'jet', 'gray', 'gray', 'gray'])
    #
    # # plot HLS gradient thresholds
    # cip.show_images(figsize=(16, 7), rows=2, fig_title='HLS x-gradients',
    #                 images=[cip.norm_abs_gradient(h_gradients_x), cip.norm_abs_gradient(l_gradients_x), cip.norm_abs_gradient(s_gradients_x),
    #                         img_binary_h_gradient_x, img_binary_l_gradient_x, img_binary_s_gradient_x],
    #                 titles=['Hue x-gradient', 'Lightness x-gradient', 'Saturation x-gradient',
    #                         'Thresholded Hue x-gradient {:}'.format(threshold_h_gradient_x),
    #                         'Thresholded Lightness x-gradient {:}'.format(threshold_l_gradient_x),
    #                         'Thresholded Saturation x-gradient {:}'.format(threshold_s_gradient_x)],
    #                 cmaps=['jet', 'jet', 'jet', 'gray', 'gray', 'gray'])
    #
    # cip.show_images(figsize=(16, 7), rows=2, fig_title='HLS y-gradients',
    #                 images=[cip.norm_abs_gradient(h_gradients_y), cip.norm_abs_gradient(l_gradients_y), cip.norm_abs_gradient(s_gradients_y),
    #                         img_binary_h_gradient_y, img_binary_l_gradient_y, img_binary_s_gradient_y],
    #                 titles=['Hue y-gradient', 'Lightness y-gradient', 'Saturation y-gradient',
    #                         'Thresholded Hue y-gradient {:}'.format(threshold_h_gradient_y),
    #                         'Thresholded Lightness y-gradient {:}'.format(threshold_l_gradient_y),
    #                         'Thresholded Saturation y-gradient {:}'.format(threshold_s_gradient_y)],
    #                 cmaps=['jet', 'jet', 'jet', 'gray', 'gray', 'gray'])

    #img_binary_magnitude = cip.magnitude_threshold(h_gradients_x, h_gradients_y, threshold=threshold_magnitude)
    #img_binary_direction = cip.direction_threshold(h_gradients_x, h_gradients_y, threshold=threshold_direction)

    #img_binary_combined = np.zeros_like(img_binary_gradient_x)
    #img_binary_combined[((img_binary_gradient_x == 1) & (img_binary_gradient_y == 1)) |
    #                    ((img_binary_magnitude == 1) & (img_binary_direction == 1))] = 1

    # plot debug data if enabled
    #if self.debug_threshold_methods:
    #    self.show_image(magnitude, title='Magnitude [0..255]', cmap='jet')
    # plot debug data if enabled
    #if self.debug_threshold_methods:
    #    self.show_image(angles, title='Gradient Direction (Angle) Image [0..1.5708]', cmap='jet')

    #
    # plot RGB and HLS channels
    #

    #
    # plot gradients and thresholds
    #
    # fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    # fig.tight_layout()
    #
    # img_color_binary = np.dstack((np.zeros_like(img_binary_direction), img_binary_gradient_x, img_binary_gradient_y))
    #
    # ax[0][0].imshow(img_rgb)
    # ax[0][1].imshow(img_binary_gradient_x, cmap='gray')
    # ax[0][2].imshow(img_binary_gradient_y, cmap='gray')
    # ax[0][3].imshow(img_color_binary)
    # ax[0][0].set_title('Original Image')
    # ax[0][1].set_title('Thresholded x-Gradient {:}'.format(threshold_gradient_x))
    # ax[0][2].set_title('Thresholded y-Gradient {:}'.format(threshold_gradient_y))
    # ax[0][3].set_title('Combined x-/y-Gradient')
    #
    # img_color_binary = np.dstack((np.zeros_like(img_binary_direction), img_binary_direction, img_binary_magnitude))
    #
    # ax[1][0].imshow(img_binary_combined, cmap='gray')
    # ax[1][1].imshow(img_binary_magnitude, cmap='gray')
    # ax[1][2].imshow(img_binary_direction, cmap='gray')
    # ax[1][3].imshow(img_color_binary)
    # ax[1][0].set_title('Combined Thresholds')
    # ax[1][1].set_title('Thresholded Magnitude {:}'.format(threshold_magnitude))
    # ax[1][2].set_title('Thresholded Direction ({:.3f}, {:.3f})'.format(threshold_direction[0], threshold_direction[1]))
    # ax[1][3].set_title('Combined Magnitude/Direction')
    #
    # fig.suptitle('Image Pre-processing Results')
    # plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.)

    plt.draw()
    plt.pause(1e-3)
    plt.show()
