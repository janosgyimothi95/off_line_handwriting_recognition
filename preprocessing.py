""" PREPROCESSING MODULE
This module's purpose is to generate a preprocessed image from the "raw" source image.
"""

from utils import *

"""
Global constants
"""
MIN_RUST_COLOR = (0, 65, 0)
MAX_RUST_COLOR = (20, 255, 255)
GAUSSIAN_KERNEL_SIZE = 45
MAXIMA_THRESHOLD = 5.0
CONTOUR_SIGNIFICANCY_TH = 150


"""
I. Rust elimination
"""
def rust_detection(source_image, show_result=True):
    ''' Detects rust points.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                rust points
    '''

    result = np.copy(source_image)


    hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    rust_mask = cv2.inRange(hsv_image, MIN_RUST_COLOR, MAX_RUST_COLOR)
    detected_rust_points = np.nonzero(rust_mask)


    if show_result:
        result[detected_rust_points] = (0, 255, 0)
        cv2.namedWindow('Detected rust', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected rust', rust_mask)
        cv2.waitKey(0)
        cv2.imshow('Detected rust', result)
        cv2.waitKey(0)

    return rust_mask



def smooth_rust_mask(rust_mask, show_result=True):
    ''' Smooths rust mask.

    :param rust_mask:       rust mask
    :param show_result:     if True shows results
    :return:                smoothed rust mask
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closed = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel, iterations=3)


    if show_result:
        cv2.namedWindow('Smooth rust', cv2.WINDOW_NORMAL)
        cv2.imshow('Smooth rust', closed)
        cv2.waitKey(0)

    return closed



def eliminate_rust_points(source_image, rust_mask, show_result=False):
    ''' Applies rust mask to image.

    :param source_image:    source image
    :param rust_mask:       rust mask
    :return:                masked image
    '''

    result = np.copy(source_image)
    color = determine_background_color(result, False)

    result[np.nonzero(rust_mask)] = color

    if show_result:
        show_image(result, 'Eliminated rust')

    return result



"""
II. Binarization
"""
def convert_image_to_grayscale(source_image, show_result=True):
    ''' Converts image to grayscale.

    :param source_image:    source image
    :return:                grayscale image
    '''

    grayscale_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    if show_result:
        cv2.namedWindow('Grayscale image', cv2.WINDOW_NORMAL)
        cv2.imshow('Grayscale image', grayscale_image)
        cv2.waitKey(0)

    return grayscale_image



def determine_binary_threshold(source_image, show_result=False):
    ''' Calculates binary threshold.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                plot of th determination
    '''

    blurred = cv2.GaussianBlur(source_image, (GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE), 0)
    threshold, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    if show_result:
        plt.clf()
        plt.hist(source_image.ravel(), bins=[i for i in range(256)], color='gray')
        plt.axvline(x=threshold, color='r', linestyle='--', label='Otsu threshold')
        plt.legend()
        plt.yticks([])
        plt.xlim([0, 256])
        plt.xticks(np.arange(0, 256, step=20))
        plt.show()

    return threshold



def binarize_grayscale_image(source_image, threshold, show_result=True):
    ''' Converts grayscale to binary.

    :param source_image:    source image
    :param threshold:       binary threshold
    :param show_result:     if True shows results
    :return:                binary image
    '''

    _, binary_image = cv2.threshold(source_image, threshold, 255, cv2.THRESH_BINARY_INV)

    if show_result:
        show_image(binary_image, 'Binary image')

    return binary_image




"""
III. ROI selection
"""
def remove_document_identifiers(source_image, show_result=True):
    ''' Removes document identifiers.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                masked document identifiers
    '''

    MORPH_CLOSE_KERNEL_SIZE = (151, 15)

    # Morph Close
    closed_image = cv2.morphologyEx(source_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KERNEL_SIZE))

    # Contour list selection
    contours = get_significant_contours(closed_image, CONTOUR_SIGNIFICANCY_TH)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])


    mean_row_intensity_vector = np.average(closed_image, axis=1)
    smoothed_vector = smooth(mean_row_intensity_vector, window_len=101, window='hamming')
    bool_maxima_vector = np.r_[True, smoothed_vector[1:] > smoothed_vector[:-1]] & np.r_[
        smoothed_vector[:-1] > smoothed_vector[1:], True]

    local_maxima_indices = [i for i, bool_value in enumerate(bool_maxima_vector) if
                            (bool_value and smoothed_vector[i] > MAXIMA_THRESHOLD and i > 100)]

    first_gap_position = int((local_maxima_indices[0] + local_maxima_indices[1]) / 2)


    potentian_boxes = sort_bounding_boxes_by_position(boundRect, first_gap_position)
    filtered_boxes = []
    for box in potentian_boxes:
        if (box[2] > 2* box[3] and box[3] < 40) or box[2] > source_image.shape[0] * 3/4:
            pass
        else:
            filtered_boxes.append(box)


    # Horizontal sorting of bounding boxes
    sorted_boxes = sorted(filtered_boxes, key=itemgetter(0))
    if len(sorted_boxes) > 1:
        selected_boxes = [sorted_boxes[0], sorted_boxes[-1]]
    elif len(sorted_boxes) == 1:
        selected_boxes = [sorted_boxes[0]]
    else:
        selected_boxes = [None]


    drawing = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)

    drawing[source_image > 0] = (255, 255, 255)

    mask = np.zeros(shape=source_image.shape, dtype=np.uint8)

    for box in sorted_boxes:
        if box is not None:
            cv2.rectangle(drawing, (int(box[0]), int(box[1])),
                      (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255), 3)

    for box in selected_boxes:
        if box is not None:
            cv2.rectangle(drawing, (int(box[0]), int(box[1])),
                      (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 3)

            cv2.rectangle(mask, (int(box[0]), int(box[1])),
                      (int(box[0] + box[2]), int(box[1] + box[3])), 255, cv2.FILLED)


    if show_result:
        show_image(drawing, 'Selected contours')
        # show_image(cv2.bitwise_and(source_image, mask), 'Masked image')

    # Generate Mask
    mask = cv2.bitwise_not(mask)

    return mask, selected_boxes



def select_roi(source_image, show_result=True):
    ''' ROI selection.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                selected ROI
    '''

    MORPH_OP_KERNEL_SIZE = (5, 5)

    opened_image = cv2.morphologyEx(source_image, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OP_KERNEL_SIZE))
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OP_KERNEL_SIZE))
    closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OP_KERNEL_SIZE))
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OP_KERNEL_SIZE))


    contours = get_significant_contours(opened_image, CONTOUR_SIGNIFICANCY_TH)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    center_points = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        center_points[i] = (int(boundRect[i][0]) + int(boundRect[i][2] / 2), int(boundRect[i][1]) + int(boundRect[i][3] / 2))

    coh_dir = create_coherency_dictionary(center_points, 400)

    to_remove = select_coherent_bounding_box_centers(coh_dir, len(center_points))

    contours = eliminate_elements_from_list(contours, to_remove)

    joined_contour = contours[0]

    for index in range(1, len(contours)):
        joined_contour = np.concatenate((joined_contour, contours[index]))


    bounding_rectangle = cv2.boundingRect(cv2.approxPolyDP(joined_contour, 3, True))

    result = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)
    result[opened_image > 0] = (255,255,255)
    if show_result:
        for i, box in enumerate(boundRect):
            if i in to_remove:
                        cv2.rectangle(result, (int(box[0]), int(box[1])),
                                                        (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255), 3)
            else:
                cv2.rectangle(result, (int(box[0]), int(box[1])),
                                                    (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 3)
        cv2.rectangle(result, (int(bounding_rectangle[0]), int(bounding_rectangle[1])),
                      (int(bounding_rectangle[0] + bounding_rectangle[2]), int(bounding_rectangle[1] + bounding_rectangle[3])), (255, 0, 0), 3)

        show_image(result, 'Selected ROI')

    return bounding_rectangle



def crop_roi(source_image, roi, show_result=True):
    ''' Crops selected ROI from image.

    :param source_image:    source image
    :param roi:             ROI
    :param show_result:     if True shows results
    :return:                cropped image
    '''

    roi = crop_rectangle_from_image(source_image, roi)

    if show_result:
        show_image(roi, 'Cropped ROI')

    return roi



"""
IV. Rotation
"""
def rotate_image(source_image, show_result=True):
    ''' Rotates image.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                rotated iamge
    '''


    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape
    MORPH_OPEN_KERNEL_SIZE = (51, 3)
    MIN_NUM_OF_VOTES = 300
    MIN_LINE_LENGTH = NUM_OF_COLS/3
    MAX_LINE_GAP = NUM_OF_COLS/4

    #Prepare image for hough transform
    opened_image = cv2.morphologyEx(source_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN_KERNEL_SIZE))

    #Preform Hough line transform
    detected_lines = cv2.HoughLinesP(opened_image, 1, np.pi / 180, MIN_NUM_OF_VOTES, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    if detected_lines is not None:

        #Calculate and filter angles
        angles = []
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            angle = np.arctan(dy / dx) * 180 / np.pi
            angles.append(angle)


        ang_mean = np.mean(angles)
        ang_med = np.median(angles)
        ang_std = np.std(angles)

        # print('Mean angle: {:4.2f}\nMedian angle: {:4.2f}\nDeviation: {:4.2f}'.format(ang_mean, ang_med, ang_std))


        if abs(ang_med) > 0.2:
            rotation_angle = ang_med
            rotated = rotate_bound(source_image, -rotation_angle)

            if show_result:
                show_image(rotated, 'Rotated image')

            return rotated

    if show_result:
        show_image(source_image, 'Rotated image')

    return source_image



"""
V. Underline removal
"""
def detect_underlines(source_image, show_result=True):
    ''' Detects underlines on image.

    :param source_image:    source image
    :param show_result:     if True shows results
    :return:                detected underlines
    '''

    GAP_AVG_INTENSITY_TH = 10
    GAP_SIZE = 20

    final_lines = []
    detected_lines = cv2.HoughLinesP(source_image, 1, np.pi / 180, 20, None,
                                     300, 10)

    to_draw = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)
    to_draw[source_image > 0] = (255, 255, 255)



    if detected_lines is not None:

        for i in range(0, len(detected_lines)):

            l = detected_lines[i][0]

            x1, y1, x2, y2 = l
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                angle = 0
            else:
                angle = np.arctan(dy / dx) * 180 / np.pi

            cv2.line(to_draw, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 5, cv2.LINE_AA)
            cv2.circle(to_draw, (l[0], l[1]), 7, (0, 255, 0), -1)
            cv2.circle(to_draw, (l[2], l[3]), 7, (0, 255, 0), -1)
            #Verifying detected lines
            roi = source_image[int((l[1]+l[3])/2) - GAP_SIZE:int((l[1]+l[3])/2) + GAP_SIZE, l[0]:l[2]]

            if abs(angle) > 0.2:
                roi = rotate_bound(roi, -angle)


            mean_int_vect = np.average(roi, axis=1)

            midpoint = int(roi.shape[0]/2)
            if (mean_int_vect[midpoint-15:midpoint] < GAP_AVG_INTENSITY_TH).any() and (
                    mean_int_vect[midpoint:midpoint+15] < GAP_AVG_INTENSITY_TH).any():
                #Collecting verified lines
                cv2.line(to_draw, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 5, cv2.LINE_AA)
                final_lines.append([l])

    if show_result:
        show_image(to_draw, 'Detected lines')


    if len(final_lines) > 0:
        return final_lines

    return None



def mask_detected_underlines(source_image, detected_lines, show_result=True):
    ''' Masks detected underlines.

    :param source_image:    source image
    :param detected_lines:  detected underline list
    :param show_result:     if True shows results
    :return:                masked underlines image
    '''

    if detected_lines is not None:

        for i in range(0, len(detected_lines)):

            l = detected_lines[i][0]
            cv2.line(source_image, (l[0], l[1]), (l[2], l[3]), 0, 25, cv2.LINE_AA)

    if show_result:
        show_image(source_image, 'Detected lines')

    return source_image



"""
Wrapper
"""
def run_preprocessing(source_image, show_subresults=False):
    ''' Runs preprocessing on given image.

    :param source_image:        source image
    :param show_subresults:     if True shows subresults
    :return:                    preprocessed image
    '''

    # RUST REMOVAL
    rust_mask = rust_detection(source_image, False)
    smooth_mask = smooth_rust_mask(rust_mask, False)
    rustless = eliminate_rust_points(source_image, smooth_mask, show_subresults)

    # BINARIZATION
    gray_rustless = convert_image_to_grayscale(rustless, False)
    th = determine_binary_threshold(gray_rustless, False)
    binary_rustless = binarize_grayscale_image(gray_rustless, th, show_subresults)

    # DOCUMENT IDENTIFIER REMOVAL
    mask, boxes = remove_document_identifiers(binary_rustless, False)
    masked_image = cv2.bitwise_and(mask, binary_rustless)

    # ROI SELECTION AND CROPPING
    roi = select_roi(cv2.bitwise_and(mask, binary_rustless), show_subresults)
    cropped = crop_roi(masked_image, roi, show_subresults)

    # ROI ROTATION
    rotated = rotate_image(cropped, show_subresults)

    # UNDERLINE ELIMINATION
    underlines = detect_underlines(rotated, False)
    preprocessed = mask_detected_underlines(rotated, underlines, show_subresults)

    show_image(preprocessed, 'Preprocessed image')

    return preprocessed