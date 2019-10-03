from utils import *


def word_segmentation(source_image, show_results=True):

    # Getting dimensions of img
    (NUM_OF_ROWS, NUM_OF_COLS) = source_image.shape

    # Printing some information about img
    print_picture_informations(source_image)

    # Generating Otsu segmented img
    ret, otsu_segmented = cv2.threshold(source_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert binary img
    inverted_colors = cv2.bitwise_not(otsu_segmented)

    # Applying morphological closing with constant kernel dimensions - TODO: Determine kernel dimensions automaticly by analyzing the image
    closed_img = cv2.morphologyEx(inverted_colors, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        45, 3)))

    # Contour extraction
    contours, _ = cv2.findContours(closed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Dismiss insignificant contours - TODO: Determine contour significancy threshold automaticly by analyzing the img
    contour_significancy_th = 70
    contours = [cnt for cnt in contours if len(cnt) > contour_significancy_th]

    # Calculate bounding rectangles
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    # Create RGB copy of source img to draw to
    drawing = cv2.cvtColor(np.copy(source_image), cv2.COLOR_GRAY2BGR)

    # Calculate and smooth mean row intensity vector TODO: Calculate convolution window size automaticly by analyzing the img
    mean_row_intensity_vector = np.average(inverted_colors, axis=1)

    smoothed_vector = smooth(mean_row_intensity_vector, window_len=101, window='hamming')

    # Determine local maxima indices of intensity vector
    bool_maxima_vector = np.r_[True, smoothed_vector[1:] > smoothed_vector[:-1]] & np.r_[
        smoothed_vector[:-1] > smoothed_vector[1:], True]

    local_maxima_indices = [i for i, bool_value in enumerate(bool_maxima_vector) if bool_value]

    # Determine maximum height of bounding rectangles TODO: Calculate threshold automaticly by analyzing the img
    height_th = int((local_maxima_indices[3] - local_maxima_indices[2]) * 1.5)

    # Eliminate too high rectangles
    new_bounding_rect_list = eliminate_double_selections(boundRect, height_th)

    # Create and populate dicitionary for {row - [segmented words]} pairs
    word_segment_dictionary = populate_word_segment_dictionary(local_maxima_indices, new_bounding_rect_list)

    if show_results:

        # Draw bounding rectangles on RGB img
        for row_index, row_value in enumerate(local_maxima_indices):
            color = (255, 0, 0)
            if word_segment_dictionary[row_value] is not None:
                for rect_index, rect in enumerate(word_segment_dictionary[row_value]):
                    cv2.putText(drawing, 'L{}W{}'.format(row_index + 1, rect_index + 1), (int(rect[0]), int(rect[1]) - 10),
                                color=color,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=4)
                    cv2.rectangle(drawing, (int(rect[0]), int(rect[1])),
                                  (int(rect[0] + rect[2]), int(rect[1] + rect[3])), color, 5)

        # Create resizeable window
        cv2.namedWindow('Word segments', cv2.WINDOW_NORMAL)

        # Resize img
        resized = resize_image(drawing)

        # Display
        cv2.imshow('Word segments', resized)
        cv2.waitKey(0)



    """ Display of row-wise average analyzis

    x = [i for i in range(NUM_OF_ROWS)]
    plt.suptitle('Row-wise average analysis of binary handwritten image')
    plt.xlim(0, len(mean_row_intensity_vector))
    plt.plot(x, mean_row_intensity_vector, color='g', alpha=0.75, label='Original signal')
    plt.plot(x, smoothed_vector, color='b', label='Smoothed signal (Hamming, k_size=101)')
    for index, max_point in enumerate(local_maxima_indices):
        plt.axvline(x=max_point, color='r', linestyle='dashed', linewidth=1, label=('Local maxima place' if index == 0 else ''))
    plt.xlabel('Serial number of row')
    plt.ylabel('Average intensity value')
    plt.legend(loc='upper left')
    plt.show()
    """


    """ Display of baslines 

    to_print = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

    for point in local_minima_indices:
        cv2.line(to_print,(0, point), (NUM_OF_COLS - 1, point),  color=(255, 0, 0), thickness=10)

    plt.subplot(221)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 'BrBG')
    plt.subplot(222)
    plt.imshow(otsu_segmented, 'gray')
    plt.subplot(223)
    plt.imshow(inverted_colors, 'gray')
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(to_print, cv2.COLOR_BGR2RGB), 'BrBG')
    plt.tight_layout()
    plt.show()
    """


    """ Plot column-wise mean intensity vectors for each row, with minima points

    y = [j for j in range(NUM_OF_COLS)]

    line_segments = []

    for base_point_index in range(len(local_minima_indices)-1):
        x = np.average(inverted_colors[local_minima_indices[base_point_index]:local_minima_indices[base_point_index+1], :], axis=0)

        indices = [i for i in range(50, len(x) - 50) if
                   (x[i] == 0 and ((x[i - 1] > x[i] and 0 == np.sum(x[i+1:i+50])) or (x[i + 1] > x[i] and 0 == sum(x[i-50:i]))))]


        plt.plot(y, x)
        for i in indices:
            plt.axvline(x=i, color='r')
        plt.show()
    """


    return word_segment_dictionary