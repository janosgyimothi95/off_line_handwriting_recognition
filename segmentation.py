""" SEGMENTATION MODULE
This module's purpose is to segment the words on the preprocessed image.
"""

from utils import *


"""
I. Separating line calculation
"""
def calculate_separating_line_list(source_image, PARTITION_NUMBER=9, show_result=True):
    ''' Calculates separating lines between each line of image.

    :param source_image:        source image
    :param PARTITION_NUMBER:    number of image partitions
    :param show_result:         if True shows results
    :return:                    separating line list
    '''

    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape

    MINIMA_THRESHOLD_VAL = 5
    COHERENT_POINTS_THRESHOLD = 100

    mean_row_intensity_vector = np.average(source_image, axis=1)
    smoothed_vector = smooth(mean_row_intensity_vector, window_len=101, window='hamming')
    bool_maxima_vector = np.r_[True, smoothed_vector[1:] > smoothed_vector[:-1]] & np.r_[
        smoothed_vector[:-1] > smoothed_vector[1:], True]

    local_maxima_indices = [i for i, bool_value in enumerate(bool_maxima_vector) if bool_value]

    line_diff_vector = [abs(x - y) for x, y in zip(local_maxima_indices[0::], local_maxima_indices[1::])]

    avg_line_diff = np.mean(line_diff_vector)

    interval = int(NUM_OF_COLS / PARTITION_NUMBER)
    col_interval = int(interval / 2)

    point_dict = {key: [] for key in range(PARTITION_NUMBER)}




    for i in range(PARTITION_NUMBER):
        if i == PARTITION_NUMBER - 1:
            roi = source_image[:, 0 + i * interval:]
        else:
            roi = source_image[:, 0 + i * interval:(i + 1) * interval]

        mean_row_intensity_vector = np.average(roi, axis=1)
        smoothed_vector = smooth(mean_row_intensity_vector, window_len=101, window='hamming')



        bool_minima_vector_1 = np.r_[True, smoothed_vector[1:] <= smoothed_vector[:-1]] & np.r_[
            smoothed_vector[:-1] < smoothed_vector[1:], True]
        bool_minima_vector_2 = np.r_[True, smoothed_vector[1:] < smoothed_vector[:-1]] & np.r_[
            smoothed_vector[:-1] <= smoothed_vector[1:], True]

        bool_minima_vector = [(bool_1 or bool_2) for bool_1, bool_2 in
                              zip(bool_minima_vector_1, bool_minima_vector_2)]
        local_minima_indices = [i for i, bool_value in enumerate(bool_minima_vector) if bool_value]
        local_minima_indices.sort()
        to_remove = []



        for a, b in zip(local_minima_indices[0::], local_minima_indices[1::]):
            if abs(b - a) < (avg_line_diff / 2):
                to_remove.append(a)
                to_remove.append(b)

        if len(to_remove) > 0:
            to_remove.sort()
            for x, y in zip(to_remove[0::2], to_remove[1::2]):
                if x in local_minima_indices: local_minima_indices.remove(x)
                if y in local_minima_indices: local_minima_indices.remove(y)
                local_minima_indices.append(int((x + y) / 2))

        local_minima_indices.sort()
        if len(local_minima_indices) > 0:
            for p in local_minima_indices:
                point_dict[i].append(p)

    starting_key = max(point_dict, key=lambda x: len(point_dict[x]))

    lower_indices = [x for x in range(starting_key - 1, -1, -1)]
    upper_indices = [x for x in range(starting_key + 1, PARTITION_NUMBER)]




    line_list = [[(col_interval + starting_key * interval, p)] for p in sorted(point_dict[starting_key])]

    if lower_indices:
        for current_key in lower_indices:
            current_col_number = col_interval + current_key * interval

            for min_point in point_dict[current_key]:

                for line in line_list:

                    if (line[0][0] == current_col_number + interval or line[0][0] == current_col_number + 2 * interval) \
                            and abs(line[0][1] - min_point) < COHERENT_POINTS_THRESHOLD:
                        line.insert(0, (current_col_number, min_point))
                        break

    if upper_indices:
        for current_key in upper_indices:
            current_col_number = col_interval + current_key * interval

            for min_point in point_dict[current_key]:

                for line in line_list:

                    if (line[-1][0] == current_col_number - interval or line[-1][
                        0] == current_col_number - 2 * interval) \
                            and abs(line[-1][1] - min_point) < COHERENT_POINTS_THRESHOLD:
                        line.append((current_col_number, min_point))
                        break



    for line_index, line in enumerate(line_list):
        if len(line) < PARTITION_NUMBER:
            left_most_point = line[0]
            right_most_point = line[-1]

            if line_index > 0:
                upper_line = line_list[line_index - 1]

                for point_index, point in enumerate(upper_line):
                    if (point[0] == left_most_point[0] - interval or point[0] == left_most_point[0] - 2 * interval) and \
                            abs(point[1] - left_most_point[1]) < COHERENT_POINTS_THRESHOLD * 1.4:
                        for i in range(point_index, -1, -1):
                            line.insert(0, upper_line[i])
                        break

                for point_index, point in enumerate(upper_line):
                    if (point[0] == right_most_point[0] + interval or point[0] == right_most_point[0] + 2 * interval) and\
                            abs(point[1] - right_most_point[1]) < COHERENT_POINTS_THRESHOLD * 1.4:
                        for i in range(point_index, len(upper_line)):
                            line.append(upper_line[i])
                        break


            left_most_point = line[0]
            right_most_point = line[-1]
            if line_index < len(line_list) - 1:
                lower_line = line_list[line_index + 1]

                for point_index, point in enumerate(lower_line):
                    if (point[0] == left_most_point[0] - interval or point[0] == left_most_point[0] - 2 * interval) and\
                            abs(point[1] - left_most_point[1]) < COHERENT_POINTS_THRESHOLD * 1.4:
                        for i in range(point_index, -1, -1):
                            line.insert(0, lower_line[i])
                        break

                for point_index, point in enumerate(lower_line):
                    if (point[0] == right_most_point[0] + interval or point[0] == right_most_point[0] + 2 * interval) and\
                            abs(point[1] - right_most_point[1]) < COHERENT_POINTS_THRESHOLD * 1.4:
                        for i in range(point_index, len(lower_line)):
                            line.append(lower_line[i])
                        break

    del line_list[-1]
    del line_list[0]

    for line in line_list:
        line.insert(0, (0, line[0][1]))
        line.append((NUM_OF_COLS, line[-1][1]))

    if show_result:
        to_draw = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)
        to_draw[source_image > 0] = (255, 255, 255)

        for line_index, line in enumerate(line_list):
            cv2.putText(to_draw, '{}'.format(line_index + 1), (10, line[0][1] - 30), color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=4)

            for a in range(len(line)):
                if line[a][0] == starting_key * interval + col_interval:
                    cv2.circle(to_draw, line[a], 15, (255, 0, 0), -1)
                else:
                    cv2.circle(to_draw, line[a], 15, (0, 255, 0), -1)
                if a < len(line) - 1:
                    pass
                    cv2.line(to_draw, line[a], line[a + 1], (0, 0, 255), 5, cv2.LINE_AA)

        show_image(to_draw, 'Separating lines')

    return line_list



"""
II. Line segmentation
"""
def get_limits_of_projection(projection_list):
    ''' Calculate lower and upper limit of horizontal projection.

    :param projection_list: projection
    :return:                lower and upper bound indices
    '''

    result = []
    for i in range(len(projection_list)):
        if projection_list[i] > 0.0:
            result.append(i)
            break

    for j in range(len(projection_list) - 1, -1, -1):
        if projection_list[j] > 0.0:
            result.append(j)
            break

    return result



def get_extreme_point_of_line(line, get_max=True):
    ''' Calculates extreme point of line.

    :param line:    line
    :param get_max: if True calculates max else calculates min point
    :return:        extreme point
    '''

    if get_max:
        return max(line, key=lambda l: l[1])[1]
    else:
        return min(line, key=lambda l: l[1])[1]



def create_segmented_line_list(source_image, line_list, show_result=True):
    ''' Creates masked, segmented line list.

    :param source_image:    source image
    :param line_list:       list od separating lines
    :param show_result:     if True shows results
    :return:                segmented line list
    '''

    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape
    first_line = [(0, 0), (NUM_OF_COLS, 0)]
    last_line = [(0, NUM_OF_ROWS), (NUM_OF_COLS, NUM_OF_ROWS)]

    line_list.insert(0, first_line)
    line_list.append(last_line)

    segments = []

    if line_list:
        for line_index in range(1, len(line_list)):

            current_line = line_list[line_index]
            prev_line = line_list[line_index - 1]
            upper_bound = get_extreme_point_of_line(prev_line, False)
            lower_bound = get_extreme_point_of_line(current_line, True)

            segments.append([upper_bound, lower_bound, [(x, y - upper_bound) for x, y in prev_line],
                             [(x, y - upper_bound) for x, y in current_line]])


    final_line_segment_list = []

    for segment_index, current_segment in enumerate(segments):
        tmp = np.copy(source_image[current_segment[0]:current_segment[1], :])


        current_segment[2].append((NUM_OF_COLS, 0))
        current_segment[2].append((0, 0))
        current_segment[3].append((NUM_OF_COLS, current_segment[1] - current_segment[0]))
        current_segment[3].append((0, current_segment[1] - current_segment[0]))

        cv2.fillPoly(tmp,
                     [np.array(current_segment[3]).astype('int32'), np.array(current_segment[2]).astype('int32')],
                     0, lineType=8)


        tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        intensity_projection_list = np.sum(tmp, axis=1)
        bounding_points = get_limits_of_projection(intensity_projection_list.tolist())


        if bounding_points and np.sum(intensity_projection_list) > 1000 * 255:
            tmp = tmp[bounding_points[0]:bounding_points[1], :]

            final_line_segment_list.append(tmp)

        if show_result:
            cv2.destroyAllWindows()
            cv2.imshow('{}. Segmetated line'.format(segment_index + 1), resize_image(tmp))
            cv2.waitKey(0)

    return final_line_segment_list



"""
III. Word segmentation
"""
def determine_word_segments(source_image, segmented_lines, show_result=True):
    ''' Creates segmented word list & final image.

    :param source_image:        source image
    :param segmented_lines:     segmented line list
    :param show_result:         if True shows results
    :return:                    result image & segmented word list
    '''

    LINE_GAP = 5
    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape

    height = -LINE_GAP
    for s in segmented_lines:
        height += s.shape[0] + LINE_GAP

    result_image = np.zeros(shape=(height, NUM_OF_COLS), dtype=np.uint8)
    height_index = 0

    word_segment_list = []

    for line_index, line in enumerate(segmented_lines):

        to_morph = np.copy(segmented_lines[line_index])
        projection = np.average(to_morph, axis=0)

        minima_lines = []
        counter = 0
        for i in range(1, len(projection) - 1):

            if projection[i] == 0 and projection[i] == projection[i - 1] and projection[i] == projection[i + 1]:
                counter += 1

            else:
                minima_lines.append(counter)
                counter = 0

        minima_lines = [x for x in minima_lines[1:] if x > 40 and x < 150]

        if minima_lines:
            kernel_size = int(np.average(minima_lines) / 5 * 3)
        else:
            kernel_size = 65


        tmp = cv2.cvtColor(segmented_lines[line_index], cv2.COLOR_GRAY2BGR)

        current_height = tmp.shape[0]

        closed = cv2.morphologyEx(to_morph, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))

        contours = get_significant_contours(closed, 130)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)


        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])


        boundRect = eliminate_inner_selections(boundRect)
        word_segments = []
        for box_index, box in enumerate(boundRect):
            word_segments.append([int(box[0]), int(box[1]) + height_index, int(box[2]), int(box[3])])

        word_segment_list.append(word_segments)

        result_image[height_index:height_index + current_height, :] = segmented_lines[line_index]
        height_index += current_height + LINE_GAP

    if show_result:
        to_draw = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

        for word_segments in word_segment_list:
            for box in word_segments:
                cv2.rectangle(to_draw, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 5)

        show_image(to_draw, 'Segmented words')

    return result_image, word_segment_list



"""
Wrapper
"""
def run_segmentation(source_image, show_subresults=False):
    ''' Runs segmentation module on given image.

    :param source_image:        source image
    :param show_subresults:     if True shows subresults
    :return:                    -
    '''

    # SEPARATING LINE GENERATION
    line_list = calculate_separating_line_list(source_image, 9, show_subresults)

    # LINE SEGMENTATION
    segmented_lines = create_segmented_line_list(source_image, line_list, show_result=show_subresults)

    # WORD SEGMENTATION
    _, _ = determine_word_segments(source_image, segmented_lines, show_result=True)