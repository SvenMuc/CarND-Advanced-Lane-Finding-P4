import argparse
import numpy as np
import cv2
import glob
import sys
import pickle
import matplotlib
matplotlib.use('macosx', force=True)  # does not supports all features on macos environments
#matplotlib.use('TKAgg', force=True)   # slow but stable on macosx environments
import matplotlib.pyplot as plt


class Calibration(object):
    """ Calibrates the camera image based on chessboard images. """

    calibrated = False              # If true, mtx, dist, rvecs and tvecs is valid.
    mtx = None                      # Intrinsic: Camera matrix (focal length fx and fy, optical center cx and cy)
    dist = None                     # Intrinsic: Distortion coefficients
    rvecs = None                    # Extrinsic: Rotation vector (camera position in world)
    tvecs = None                    # Extrinsic: Translation vector (camera position in world)

    def __init__(self, filename=None):
        """ Initialization method.

        :param filename: Filename of calibration file. If None, no calibration data will be restored.
        """

        if filename:
            self.load_calibration(filename)

    def calibrate_with_chessboard_files(self, path, nb_corners):
        """ Reads all chessboard images in given path.

        :param path:    Path to chessboard images.
        :param corners: Number of corners of chessboard (nb_x, nb_y)

        :return: Returns True if calibration was successful. Otherwise False.
        """

        cal_files = glob.glob(path + '/*.jpg')

        if len(cal_files) == 0:
            print('ERROR: No jpg images found in path {:s}'.format(path), file=sys.stderr)
            return False

        cnt = 0
        obj_points = []         # 3D points in real world
        img_points = []         # 2D points in image plane

        for file in cal_files:
            img_rgb = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

            # prepare object points like (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
            objp = np.zeros((nb_corners[0] * nb_corners[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:nb_corners[0], 0:nb_corners[1]].T.reshape(-1, 2)

            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(img_gray, nb_corners, None)

            if ret:
                img_points.append(corners)
                obj_points.append(objp)
                cnt += 1

                img_corners = cv2.drawChessboardCorners(img_rgb, nb_corners, corners, ret)
                cv2.imshow(file, img_corners)
                cv2.waitKey(0)
            else:
                print('ERROR: Found no chessboard corners {:} on image {:s}'.format(nb_corners, file), file=sys.stderr)

        cv2.destroyAllWindows()

        # calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)

        if ret:
            self.mtx = mtx
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.calibrated = True
        else:
            print('ERROR: Camera calibration failed.', file=sys.stderr)
            return False

        print('Calibrated successfully based on {:d}/{:d} chessboards.'.format(cnt, len(cal_files)))
        print('mtx:   {:}'.format(mtx))
        print('dist:  {:}'.format(dist))
        print('rvecs: {:}'.format(rvecs))
        print('tvecs: {:}'.format(tvecs))

        return True

    def undistort(self, image):
        """ Undistorts the image.

        :param image: Input image.

        :return: Returns the undistorted image. If calibrated is False returns the original image.
        """
        if self.calibrated:
            return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        else:
            print('ERROR: No valid calibration parameter. Returned input image.', file=sys.stderr)
            return image

    def save_calibration(self, filename):
        """ Save camera calibration to DAT file.

        :param filename: Camera calibration DAT filename.

        :return: Returns True if calibration saved successfully.
        """

        file = open(filename, 'wb')

        if file:
            data = {}
            data['calibrated'] = self.calibrated
            data['mtx'] = self.mtx
            data['dist'] = self.dist
            data['rvecs'] = self.rvecs
            data['tvecs'] = self.tvecs

            pickle.dump(data, file)
            print('Saved calibration data to {:s}'.format(filename))
            return True
        else:
            print('ERROR: Failed to save calibration to file {:s}'.format(filename), file=sys.stderr)
            return False

    def load_calibration(self, filename):
        """ Load camera calibration from DAT file.

        :param filename: Camera calibration DAT file.

        :return: Returns True if calibration loaded successfully.
        """

        file = open(filename, 'rb')

        if file:
            data = pickle.load(file)

            if data['calibrated']:
                self.calibrated = data['calibrated']
                self.mtx = data['mtx']
                self.dist = data['dist']
                self.rvecs = data['rvecs']
                self.tvecs = data['tvecs']

                print('Loaded calibration data from {:s} successfully'.format(filename))
                return True
            else:
                print('ERROR: Failed to load calibration from file {:s}'.format(filename), file=sys.stderr)
                return None
        else:
            print('ERROR: Failed to open calibration file {:s}'.format(filename), file=sys.stderr)
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration')

    parser.add_argument(
        '-c', '--calibrate',
        help='Calibrates the camera with chessboard images in PATH.',
        dest='calib_files',
        metavar='PATH'
    )

    parser.add_argument(
        '-s', '--save',
        help='Save camera calibration to file.',
        dest='save_dat_file',
        metavar='DAT_FILE'
    )

    parser.add_argument(
        '-t', '--test-undistort',
        help='Undistort test image with calibration dat file.',
        dest='load_dat_file',
        metavar='DAT_FILE'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    elif args.calib_files:
        # Calibrates the camera with given chessboard images.

        if args.save_dat_file is None:
            print('ERROR: Use -s parameter to save calibration data to file.', file=sys.stderr)
            exit(-1)

        calib = Calibration()
        ret = calib.calibrate_with_chessboard_files(args.calib_files, (9, 6))

        if ret:
            calib.save_calibration(args.save_dat_file)

    elif args.load_dat_file:
        # Test calibration
        calib = Calibration(args.load_dat_file)
        img_rgb_1 = cv2.cvtColor(cv2.imread('camera_cal/calibration1.jpg'), cv2.COLOR_BGR2RGB)
        img_rgb_2 = cv2.cvtColor(cv2.imread('test_images/straight_lines2.jpg'), cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(2, 2, figsize=(11, 6))
        plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.05)
        ax[0][0].imshow(img_rgb_1)
        ax[0][0].set_title('Distorted Image')
        ax[1][0].imshow(img_rgb_2)
        ax[1][0].set_title('Distorted Image')
        ax[0][1].imshow(calib.undistort(img_rgb_1))
        ax[0][1].set_title('Undistorted Image')
        ax[1][1].imshow(calib.undistort(img_rgb_2))
        ax[1][1].set_title('Undistorted Image')
        plt.show()
