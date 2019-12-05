""" UTILITIES MODULE
This module is a collection of utility functions/classes.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import os
import random
from operator import itemgetter
import itertools
import math


"""
Interactive classes
"""
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
        show_image('Interactive thresholding', th_image)

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



class interactive_hsv_threshold:
    '''Interactive HSV thresholding class:
    The purpose of this class is to create an interactive window in which the user can freely test out different
    thresholds of the given HSV image
    '''

    def __init__(self, image):
        self.image = image
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.lower_h = 0
        self.upper_h = 180
        self.lower_s = 0
        self.upper_s = 255
        self.lower_v = 0
        self.upper_v = 255

    def threshold_clip(self):
        copy_img = np.copy(self.image)
        hsv_mask = cv2.inRange(self.hsv_image, (self.lower_h, self.lower_s, self.lower_v), (self.upper_h, self.upper_s, self.upper_v))
        print(hsv_mask.dtype, hsv_mask.shape, np.min(hsv_mask))

        copy_img[np.nonzero(hsv_mask)] = (0,255,0)
        show_image('Interactive HSV thresholding', copy_img)


    def trackbar_lower_hue(self, th):
        self.lower_h = th
        self.threshold_clip()

    def trackbar_upper_hue(self, th):
        self.upper_h = th
        self.threshold_clip()

    def trackbar_lower_sat(self, th):
        self.lower_s = th
        self.threshold_clip()

    def trackbar_upper_sat(self, th):
        self.upper_s = th
        self.threshold_clip()

    def trackbar_lower_val(self, th):
        self.lower_v = th
        self.threshold_clip()

    def trackbar_upper_val(self, th):
        self.upper_v = th
        self.threshold_clip()

    def interactive_hsv_thresholding(self):
        self.threshold_clip()
        cv2.createTrackbar('Lower hue threshold', 'Interactive HSV thresholding', self.lower_h, 180, self.trackbar_lower_hue)
        cv2.createTrackbar('Upper hue threshold', 'Interactive HSV thresholding', self.upper_h, 180, self.trackbar_upper_hue)
        cv2.createTrackbar('Lower saturation threshold', 'Interactive HSV thresholding', self.lower_s, 255, self.trackbar_lower_sat)
        cv2.createTrackbar('Upper saturation threshold', 'Interactive HSV thresholding', self.upper_s, 255, self.trackbar_upper_sat)
        cv2.createTrackbar('Lower value threshold', 'Interactive HSV thresholding', self.lower_v, 255, self.trackbar_lower_val)
        cv2.createTrackbar('Upper value threshold', 'Interactive HSV thresholding', self.upper_v, 255, self.trackbar_upper_val)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



class interctive_morphology:
    ''' Interactive morphology class
     The purpose of this class is to create an interactive window in which the user can freely test out different
    types of morphological operators
    '''

    valid_modes = [cv2.MORPH_DILATE, cv2.MORPH_ERODE, cv2.MORPH_CLOSE, cv2.MORPH_OPEN]

    def __init__(self, image, mode):
        self.image = image
        self.mode = mode
        self.ellipse_x_size = 5
        self.ellipse_y_size = 5


    def execute_operation(self):
        print('x: {}\ny: {}'.format(self.ellipse_x_size, self.ellipse_y_size))
        morphed_image = cv2.morphologyEx(self.image, interctive_morphology.valid_modes[self.mode], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        self.ellipse_x_size, self.ellipse_y_size)))
        show_image(morphed_image, 'Interactive morphology')


    def ellipse_x_size_specifier(self, size):
        self.ellipse_x_size = size
        self.execute_operation()


    def ellipse_y_size_specifier(self, size):
        self.ellipse_y_size = size
        self.execute_operation()


    def mode_specifier(self, mode):
        self.mode = mode
        self.execute_operation()


    def interactive_morph(self):
        self.execute_operation()
        cv2.createTrackbar('Mode', 'Interactive morphology', self.mode, 3, self.mode_specifier)
        cv2.createTrackbar('Ellipse x size', 'Interactive morphology', self.ellipse_x_size, 255, self.ellipse_x_size_specifier)
        cv2.createTrackbar('Ellipse y size', 'Interactive morphology', self.ellipse_y_size, 255, self.ellipse_y_size_specifier)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



class interactive_hough_line_trnasform:
    ''' Interactive Hough line transform class
     The purpose of this class is to create an interactive window in which the user can freely test out Hough line
     transform
    '''

    def __init__(self, image):
        self.image = image
        self.min_num_of_votes = 200
        self.min_line_length = 300
        self.max_gap_length = 0


    def execute_operation(self):
        print('\nSearching Hough lines with attributes: \nMin num of votes: {}\nMin line length: {}\nMax line gap: {}'
              .format(self.min_num_of_votes, self.min_line_length, self.max_gap_length))

        to_draw = np.zeros(shape=self.image.shape + (3,), dtype=np.uint8)
        to_draw[self.image > 0] = (255, 255, 255)

        detected_lines = cv2.HoughLinesP(self.image, 1, np.pi / 180, self.min_num_of_votes, None,
                                         self.min_line_length, self.max_gap_length)

        if detected_lines is not None:
            for i in range(0, len(detected_lines)):
                l = detected_lines[i][0]
                cv2.line(to_draw, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(to_draw, (l[0], l[1]), 7, (0, 255, 0), -1)
                cv2.circle(to_draw, (l[2], l[3]), 7, (0, 255, 0), -1)

        cv2.namedWindow('Interactive Hough line transform', cv2.WINDOW_NORMAL)
        cv2.imshow('Interactive Hough line transform', to_draw)


    def vote_number_specifier(self, votes):
        self.min_num_of_votes = votes
        self.execute_operation()

    def line_length_specifier(self, min_length):
        self.min_line_length = min_length
        self.execute_operation()

    def line_gap_specifier(self, max_gap):
        self.max_gap_length = max_gap
        self.execute_operation()

    def interactive_hough(self):
        self.execute_operation()
        cv2.createTrackbar('Min number of votes', 'Interactive Hough line transform', self.min_num_of_votes, 1500,
                           self.vote_number_specifier)
        cv2.createTrackbar('Min line length', 'Interactive Hough line transform', self.min_line_length, 1500,
                           self.line_length_specifier)
        cv2.createTrackbar('Max line gap length', 'Interactive Hough line transform', self.max_gap_length, 25,
                           self.line_gap_specifier)

        cv2.waitKey(0)
        cv2.destroyAllWindows()




"""
General
"""
def smooth(x, window_len=11, window='hanning'):
    ''' Smooths the given vectors with the given windowtype, with the given size

    :param x:               vector
    :param window_len:      window size
    :param window:          window type
    :return:                smoothed vector
    '''

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2):-int(window_len / 2)]



def threshold_between_given_bounds(grayscale_image, lower_threshold, upper_threshold):
    ''' Executes a thresholding of the given image between the given bounds.

    :param grayscale_image:     input image
    :param lower_threshold:     lower threshold value
    :param upper_threshold:     upper threshold value
    :return:                    thresheld image
    '''

    th_image = np.copy(grayscale_image)
    th_image[grayscale_image > upper_threshold] = 0
    th_image[grayscale_image <= upper_threshold] = 255
    th_image[grayscale_image < lower_threshold] = 0

    return th_image



def eliminate_double_selections(rectengle_list, height_threshold):
    ''' Eliminates selections that are bigger than a given threshold.

    :param rectengle_list:      selection list
    :param height_threshold:    height threshold
    :return:                    eliminated rectangle list
    '''

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
    ''' Selects the inner selection of two rectangles.

    :param rect_a:  rectangle a
    :param rect_b:  rectangle b
    :return:        inner selection | None
    '''

    result = None

    if rect_a[0] > rect_b[0] and rect_a[1] > rect_b[1] and rect_a[2] + rect_a[0] < rect_b[2] + rect_b[0] and \
            rect_a[3] + rect_a[1] < rect_b[3] + rect_b[1]:
        result = rect_a
    elif rect_b[0] > rect_a[0] and rect_b[1] > rect_a[1] and rect_b[2] + rect_b[0] < rect_a[2] + rect_a[0] and \
            rect_b[3] + rect_b[1] < rect_a[3] + rect_a[1]:
        result = rect_b
    return result


def eliminate_inner_selections(rectangle_list):
    ''' Eliminates inner selections from rectangle list.

    :param rectangle_list:  rectangle list
    :return:                eliminated rectangel list
    '''

    to_remove = []
    for a, b in itertools.combinations(rectangle_list, 2):
        ret = get_inner_selection(a, b)
        if ret is not None:
            to_remove.append(ret)

    for inner_selection in to_remove:
        if inner_selection in rectangle_list:
            rectangle_list.remove(inner_selection)

    return rectangle_list



def populate_word_segment_dictionary(line_center_indices, bounding_rectangle_list):
    ''' Creates word segment directory.

    :param line_center_indices:     idices of line center points
    :param bounding_rectangle_list: bounding rectangle list
    :return:                        word segment directory
    '''

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



def resize_image(source_image):
    ''' Resizes image.

    :param source_image: source image
    :return:             resized image
    '''

    height, width = source_image.shape[:2]
    W = 1500
    scale = W/width
    new_x, new_y = source_image.shape[1]*scale, source_image.shape[0]*scale

    new_image = cv2.resize(source_image, (int(new_x), int(new_y)))

    return new_image


def print_picture_informations(picture):
    ''' Prints some information about the image.

    :param picture: source image
    :return:        information
    '''
    (num_of_rows, num_of_cols) = picture.shape

    print('\nPICTURE INFORMATION\nShape: ({}, {})\nDatatype:     {}\n{}'.format(num_of_rows, num_of_cols, picture.dtype,
                                                                             '___________________'))

def determine_background_color(source_image, show_result=True):
    ''' Determines most frequently occuring background color.

    :param source_image:    source image
    :param show_result:     If true shows result
    :return:                most frequent background color
    '''

    bg_color = ()
    color_labels = ['kék', 'zöld', 'piros']
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        current_channel_hist = cv2.calcHist([source_image], [i], None, [256], [0, 256])
        most_frequent_intensity = np.argmax(current_channel_hist)
        bg_color = bg_color + (most_frequent_intensity,)
        if show_result:
            plt.plot(current_channel_hist, color=col, alpha=0.75, label=color_labels[i] + ' csatorna')
            plt.axvline(x=most_frequent_intensity, color=col, linestyle='-.', label='leggyakoribb intenzitásérték a {} csatornán'.format(color_labels[i]))
            print('Maximum value on channel {}: {}'.format(col, most_frequent_intensity))
            plt.xlim([0, 256])
            plt.xticks(np.arange(0, 256, step=16))

    if show_result:
        plt.legend()
        plt.grid()
        plt.ylabel('Gyakoriság')
        plt.xlabel('Intenzitás')
        plt.suptitle("Színcsatornák hisztogramjai")
        plt.show()

    return bg_color


def plot_image_comparison(image_list, colormaps):
    ''' Plots given images.

    :param image_list:  image list
    :param colormaps:   colormap list
    :return:            plot
    '''

    plot_titles = ['(a)', '(b)', '(c)', '(d)']
    for i in range(len(image_list)):
        ax = plt.subplot(220 + i + 1)
        ax.set_title(plot_titles[i])
        if colormaps[i] == 'gray':
            plt.imshow(image_list[i], cmap=colormaps[i])
        else:
            plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.axis('tight')

    plt.tight_layout()
    plt.show()



def show_image(source_image, window_name='Examined image'):
    ''' Shows image in a named window.

    :param source_image:  source image
    :param window_name:   window name
    :return:              image in window
    '''

    cv2.destroyAllWindows()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, source_image)
    cv2.waitKey(0)



def get_significant_contours(source_image, threshold):
    ''' Extracts the significant contours of an image.

    :param source_image:    source image
    :param threshold:       significancy threshold
    :return:                conrous list
    '''

    contours, _ = cv2.findContours(source_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return [cnt for cnt in contours if cnt.shape[0] > threshold]



def sort_bounding_boxes_by_position(bounding_box_list, threshold):
    ''' Sorts bounding boxes by their horizontal position

    :param bounding_box_list:   bounding box list
    :param threshold:           height threshold
    :return:
    '''

    return[box for box in bounding_box_list if (int(box[1]) + int(box[3] / 2) < threshold)]



def create_coherency_dictionary(bounding_box_center_list, coherency_threshold):
    ''' Creates coherency dictionary

    :param bounding_box_center_list:    Bounding box center list
    :param coherency_threshold:         coherency threshold
    :return:                            coherency dictionary
    '''

    coh_dict = {}

    for a, b in itertools.combinations([i for i in range(len(bounding_box_center_list))], 2):
        coh_dict[(a,b)] = calculate_distance_between_points(bounding_box_center_list[a], bounding_box_center_list[b]) < coherency_threshold

    return coh_dict



def select_coherent_bounding_box_centers(coherency_dictionary, number_of_centers):
    ''' Selects coherent bounding boxess

    :param coherency_dictionary:    coherency dictionary
    :param number_of_centers:       number of center points
    :return:                        coherent bounding box list
    '''

    keys = coherency_dictionary.keys()
    to_eliminate = []
    potentian_coherents = []
    for i in range(number_of_centers):
        current_keys = [key for key in keys if i in key]
        num_of_trues = 0
        index = 0
        while (index < len(current_keys) and num_of_trues < 2):
            if coherency_dictionary[current_keys[index]]:

                num_of_trues += 1
            index += 1

        if num_of_trues == 1:
            potentian_coherents.append(i)
        elif num_of_trues == 0:
            to_eliminate.append(i)

    for a, b in itertools.combinations(potentian_coherents, 2):
        curr_key = [key for key in keys if (a in key and b in key)]
        if coherency_dictionary[curr_key[0]]:
            to_eliminate.append(a)
            to_eliminate.append(b)

    to_eliminate.sort(reverse=True)
    return to_eliminate



def calculate_distance_between_points(A, B):
    ''' Calculates the euqlidian distance between two given point

    :param A:   point A
    :param B:   point B
    :return:    distance between A and B
    '''
    return math.sqrt(math.pow(A[0]-B[0], 2)+ math.pow(A[1]-B[1], 2))



def eliminate_elements_from_list(examined_list, undesired_idecies):
    ''' Eliminates undesired elements from a list

    :param examined_list:       list
    :param undesired_idecies:   undesired indices
    :return:                    eliminated list
    '''

    undesired_idecies.sort(reverse=True)
    for i in undesired_idecies:
        del examined_list[i]

    return examined_list



def crop_rectangle_from_image(source_image, rectangle):
    ''' Cuts out the given rectangular region of the source image.

    :param source_image:    source image
    :param rectangle:       rectangular region
    :return:                cropped image
    '''

    result = source_image.copy()
    return result[int(rectangle[1]):int(rectangle[1] + rectangle[3] + 1), int(rectangle[0]):int(rectangle[0]+ rectangle[2] + 1)]


def rotate_bound(image, angle):
    ''' Rotates the given image with the given angle

    :param image:   image
    :param angle:   rotation angle
    :return:        rotated image
    '''

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))



def generate_horizontal_line_point(num_of_rows, num_of_lines=10):
    ''' generates equvidistant horizontal points

    :param num_of_rows:     number of rows
    :param num_of_lines:    number of lines
    :return:                point list
    '''


    line_gap = int(num_of_rows/(num_of_lines + 1))

    points = []
    for i in range(num_of_lines):
        points.append((i + 1) * line_gap)

    return points



def create_horizontal_dashed_line(image, index = 50, dash_size = 150 ,color=(255, 0,0)):
    ''' Creates horizontal dashed line with the given parameters.

    :param image:       source image
    :param index:       height
    :param dash_size:   dash size
    :param color:       color
    :return:            image with dashed line
    '''

    num_of_cols = 5000

    point_pairs = [[(i*2)*dash_size,(i*2 + 1)* dash_size] for i in range(int(num_of_cols/dash_size/2))]
    print(point_pairs)
    for p in point_pairs:
        cv2.line(image, (p[0], index), (p[1], index), color, 2, cv2.LINE_AA)



def convert_int_to_label_string(int_id):
    ''' Converts int image id to string.

    :param int_id:  int image id
    :return:        string image id
    '''

    string_id = str(int_id)
    string_id = (3-len(string_id))*'0' + string_id

    return string_id