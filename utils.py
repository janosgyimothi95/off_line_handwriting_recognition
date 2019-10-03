import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import random
from operator import itemgetter
import itertools


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len/2):-int(window_len/2)]


def threshold_between_given_bounds(grayscale_image, lower_threshold, upper_threshold):
    '''

    :param grayscale_image:
    :param lower_threshold:
    :param upper_threshold:
    :return:
    '''

    th_image = np.copy(grayscale_image)
    th_image[grayscale_image > upper_threshold] = 0
    th_image[grayscale_image <= upper_threshold] = 255
    th_image[grayscale_image < lower_threshold] = 0

    return th_image


class interactive_threshold:
    '''Interactive thresholding class:
    The purpose of this class is to create an interactive window in which the user can freely test out different
    thresholds of the given grayscale image
    '''

    def __init__(self, image):
        self.image = image
        self.lower_th = 0
        self.upper_th = 255

    def threshold_clip(self):
        th_image = threshold_between_given_bounds(self.image, self.lower_th, self.upper_th)
        cv2.imshow('Interactive thresholding', th_image)

    def trackbar_lower(self, th):
        self.lower_th = th
        self.threshold_clip()

    def trackbar_upper(self, th):
        self.upper_th = th
        self.threshold_clip()

    def interactive_thresholding(self):
        self.threshold_clip()
        cv2.createTrackbar('Lower threshold', 'Interactive thresholding', self.lower_th, 255, self.trackbar_lower)
        cv2.createTrackbar('Upper threshold', 'Interactive thresholding', self.upper_th, 255, self.trackbar_upper)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



class interctive_morphology:

    def __init__(self, image):
        self.image = image
        # self.mode = mode
        self.ellipse_x_size = 5
        self.ellipse_y_size = 5


    def execute_operation(self):
        morphed_image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        self.ellipse_x_size, self.ellipse_y_size)))
        cv2.imshow('Interactive morphology', morphed_image)


    def ellipse_x_size_specifier(self, size):
        self.ellipse_x_size = size
        self.execute_operation()

    def ellipse_y_size_specifier(self, size):
        self.ellipse_y_size = size
        self.execute_operation()


    def interactive_morph(self):
        self.execute_operation()
        cv2.createTrackbar('Ellipse x size', 'Interactive morphology', self.ellipse_x_size, 255, self.ellipse_x_size_specifier)
        cv2.createTrackbar('Ellipse y size', 'Interactive morphology', self.ellipse_y_size, 255, self.ellipse_y_size_specifier)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def eliminate_double_selections(rectengle_list, height_threshold):

    new_list = []
    for index, rect in enumerate(rectengle_list):

        if rect[3] > height_threshold:
            new_upper_rect = (rect[0], rect[1], rect[2], int(rect[3]/2))
            new_lower_rect = (rect[0], rect[1] + int(rect[3]/2), rect[2], int(rect[3]/2))

            new_list.append(new_lower_rect)
            new_list.append(new_upper_rect)
        else:
            new_list.append(rect)

    return new_list


def get_inner_selection(rect_a, rect_b):

    result = None

    if rect_a[0] > rect_b[0] and rect_a[1] > rect_b[1] and rect_a[2] + rect_a[0] < rect_b[2] + rect_b[0] and \
            rect_a[3] + rect_a[1] < rect_b[3] + rect_b[1]:
        result = rect_a
    elif rect_b[0] > rect_a[0] and rect_b[1] > rect_a[1] and rect_b[2] + rect_b[0] < rect_a[2] + rect_a[0] and \
            rect_b[3] + rect_b[1] < rect_a[3] + rect_a[1]:
        result = rect_b
    return result


def eliminate_inner_selections(rectangle_list):

    to_remove = []
    for a, b in itertools.combinations(rectangle_list, 2):
        ret = get_inner_selection(a, b)
        if ret is not None:
            to_remove.append(ret)

    for inner_selection in to_remove:
        if inner_selection in rectangle_list:
            rectangle_list.remove(inner_selection)

    return rectangle_list


def resize_image(source_image):

    height, width = source_image.shape[:2]
    W = 1500
    scale = W/width
    new_x, new_y = source_image.shape[1]*scale, source_image.shape[0]*scale

    new_image = cv2.resize(source_image, (int(new_x), int(new_y)))

    return new_image


def populate_word_segment_dictionary(line_center_indices, bounding_rectangle_list):


    #Create dicitionary for {row - [segmented words]} pairs
    word_segment_dictionary = dict((line_position, []) for line_position in line_center_indices)

    #Populate dictionary
    for rect in bounding_rectangle_list:
        for line_pos in line_center_indices:
            if rect[1] <= line_pos and rect[1]+rect[3] >= line_pos:
                word_segment_dictionary[line_pos].append(rect)

    #Sort rectengles of each key by their horizontal position
    for key in word_segment_dictionary.keys():
        word_segment_dictionary[key] = eliminate_inner_selections(sorted(word_segment_dictionary[key], key=itemgetter(0)))


    return word_segment_dictionary


def print_picture_informations(picture):
    (num_of_rows, num_of_cols) = picture.shape

    print('\nPICTURE INFORMATION\nShape: ({}, {})\nDatatype:     {}\n{}'.format(num_of_rows, num_of_cols, picture.dtype,
                                                                             '___________________'))