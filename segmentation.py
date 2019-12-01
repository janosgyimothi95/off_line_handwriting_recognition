from utils import *


def get_baselines(source_image, show_result=True):

    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape
    PARTITION_NUMBER = 5

    interval = int(NUM_OF_COLS/PARTITION_NUMBER)



    for i in range(PARTITION_NUMBER):
        if i == PARTITION_NUMBER -1:
            print([0 + i * interval, ':', NUM_OF_COLS])
        else:
            print([0 + i*interval, ':', (i+1) * interval])

"""

"""

def calculate_separating_line_list(source_image, PARTITION_NUMBER=9, show_result=True):
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

    #toprint
    vector_list = []
    picture_segments = []




    for i in range(PARTITION_NUMBER):
        if i == PARTITION_NUMBER - 1:
            roi = source_image[:, 0 + i * interval:]
        else:
            roi = source_image[:, 0 + i * interval:(i + 1) * interval]

        mean_row_intensity_vector = np.average(roi, axis=1)
        smoothed_vector = smooth(mean_row_intensity_vector, window_len=101, window='hamming')

        ''
        vector_list.append(smoothed_vector)



        bool_minima_vector_1 = np.r_[True, smoothed_vector[1:] <= smoothed_vector[:-1]] & np.r_[
            smoothed_vector[:-1] < smoothed_vector[1:], True]
        bool_minima_vector_2 = np.r_[True, smoothed_vector[1:] < smoothed_vector[:-1]] & np.r_[
            smoothed_vector[:-1] <= smoothed_vector[1:], True]

        bool_minima_vector = [(bool_1 or bool_2) for bool_1, bool_2 in
                              zip(bool_minima_vector_1, bool_minima_vector_2)]
        local_minima_indices = [i for i, bool_value in enumerate(bool_minima_vector) if bool_value]
        local_minima_indices.sort()
        to_remove = []

        ''
        to_draw_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        for p in local_minima_indices:
            cv2.circle(to_draw_roi, (col_interval, p), 7, (255, 0, 0,), -1)

        picture_segments.append(to_draw_roi)

        for a, b in zip(local_minima_indices[0::], local_minima_indices[1::]):
            # if abs(b-a) < 100:
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




    # to_draw = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)
    # to_draw[source_image == 0] = (255, 255, 255)
    #
    # for index, key in enumerate(point_dict.keys()):
    #     col = index *interval + col_interval
    #     for p in point_dict[key]:
    #
    #         cv2.circle(to_draw, (col, p), 15, (255, 0, 0), -1)
    #
    # show_image(to_draw, 'Separating lines')
    # cv2.imwrite('../../starting_points_23.png', to_draw)



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


    #TO print
    # to_draw = np.zeros(shape=source_image.shape + (3,), dtype=np.uint8)
    # to_draw[source_image == 0] = (255, 255, 255)
    #
    # for line_index, line in enumerate(line_list[1:-1]):
    #
    #     for a in range(len(line)):
    #         if line[a][0] == starting_key * interval + col_interval:
    #             cv2.circle(to_draw, line[a], 10, (0, 255, 0), -1)
    #         else:
    #             cv2.circle(to_draw, line[a], 10, (255, 0, 0), -1)
    #         if a < len(line) - 1:
    #             pass
    #             cv2.line(to_draw, line[a], line[a + 1], (0, 0, 255), 5, cv2.LINE_AA)





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
        # to_draw[source_image == 0] = (255, 255, 255)
        to_draw[source_image > 0] = (255, 255, 255)

        for line_index, line in enumerate(line_list):
            cv2.putText(to_draw, '{}'.format(line_index + 1), (10, line[0][1] - 30), color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=4)

            for a in range(len(line)):
                if line[a][0] == starting_key * interval + col_interval:
                    cv2.circle(to_draw, line[a], 10, (0, 255, 0), -1)
                else:
                    cv2.circle(to_draw, line[a], 10, (255, 0, 0), -1)
                if a < len(line) - 1:
                    pass
                    cv2.line(to_draw, line[a], line[a + 1], (0, 0, 255), 5, cv2.LINE_AA)

        # show_image(to_draw, 'Separating lines')


    # return line_list
    return line_list, picture_segments, vector_list

"""
Line segmentation
"""

def get_limits_of_projection(projection_list):
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
    if get_max:
        return max(line, key=lambda l: l[1])[1]
    else:
        return min(line, key=lambda l: l[1])[1]




def create_segmented_line_list(source_image, line_list, show_result=True):


    to_draw = cv2.cvtColor(cv2.bitwise_not(source_image), cv2.COLOR_GRAY2BGR)


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

            # if line_index == 2:
            #     cv2.line(to_draw, (0, upper_bound), (NUM_OF_COLS, upper_bound), (255, 0, 0), 5, cv2.LINE_AA)
            #     cv2.line(to_draw, (0, lower_bound), (NUM_OF_COLS, lower_bound), (255, 0, 0), 5, cv2.LINE_AA)
            #     show_image(to_draw[:750, :], 'asd')
            #     cv2.imwrite('../../initial_img_23_02.png', to_draw[:750, :])

            segments.append([upper_bound, lower_bound, [(x, y - upper_bound) for x, y in prev_line],
                             [(x, y - upper_bound) for x, y in current_line]])



    final_line_segment_list = []

    for segment_index, current_segment in enumerate(segments):
        tmp = np.copy(source_image[current_segment[0]:current_segment[1], :])
        # tmp = cv2.cvtColor(preprocessed[current_segment[0]:current_segment[1],:], cv2.COLOR_GRAY2BGR)


        to_draw_tmp = cv2.cvtColor(cv2.bitwise_not(tmp), cv2.COLOR_GRAY2BGR)




        current_segment[2].append((NUM_OF_COLS, 0))
        current_segment[2].append((0, 0))
        # current_segment[2].append(current_segment[2][0])
        current_segment[3].append((NUM_OF_COLS, current_segment[1] - current_segment[0]))
        current_segment[3].append((0, current_segment[1] - current_segment[0]))
        # current_segment[2].append(current_segment[3][0])
        # print(np.array(current_segment[2]))

        cv2.fillPoly(tmp,
                     [np.array(current_segment[3]).astype('int32'), np.array(current_segment[2]).astype('int32')],
                     0, lineType=8)

        # cv2.fillPoly(to_draw_tmp,
        #              [np.array(current_segment[3]).astype('int32'), np.array(current_segment[2]).astype('int32')],
        #              [0, 0, 255], lineType=8)

        # cv2.fillPoly(to_draw_tmp,
        #              [np.array(current_segment[3]).astype('int32'), np.array(current_segment[2]).astype('int32')],
        #              [0, 0, 0], lineType=8)

        #
        for p_index, p in enumerate(current_segment[2]):
            if p_index > len(current_segment[2]) - 2:
                cv2.circle(to_draw_tmp, p, 7, (255, 0, 0), -1)
            else:
                cv2.circle(to_draw_tmp, p, 7, (0, 255, 0), -1)
            if p_index < len(current_segment[2])-1:
                cv2.line(to_draw_tmp, p, current_segment[2][p_index + 1], (0, 0, 255), 3, cv2.LINE_AA)

        for q_index, q in enumerate(current_segment[3]):
            if q_index > len(current_segment[3]) - 2:
                cv2.circle(to_draw_tmp, q, 7, (255, 0, 0), -1)
            else:
                cv2.circle(to_draw_tmp, q, 7, (0, 255, 0), -1)
            if q_index < len(current_segment[3])-1:
                cv2.line(to_draw_tmp, q, current_segment[3][q_index + 1], (0, 0, 255), 3, cv2.LINE_AA)

        cv2.line(to_draw_tmp, current_segment[2][0], current_segment[2][-1], (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(to_draw_tmp, current_segment[3][0], current_segment[3][-1], (0, 0, 255), 3, cv2.LINE_AA)




        tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))






        intensity_projection_list = np.sum(tmp, axis=1)

        bounding_points = get_limits_of_projection(intensity_projection_list.tolist())





        # cv2.line(tmp, (0, bounding_points[0]), (NUM_OF_COLS, bounding_points[0]), 255, 1,
        #          cv2.LINE_AA)
        # cv2.line(tmp, (0, bounding_points[1]), (NUM_OF_COLS, bounding_points[1]), 255, 1,
        #          cv2.LINE_AA)

        # cv2.line(tmp, (0, current_interval[0]), (NUM_OF_COLS, current_interval[0]), (255, 0, 255), 5, cv2.LINE_AA)
        # cv2.line(tmp, (0, current_interval[1]), (NUM_OF_COLS, current_interval[1]), (255, 0, 255), 5, cv2.LINE_AA)
        # show_image(tmp[current_interval[0]:current_interval[1], :], 'segmetation lines')

        if bounding_points and np.sum(intensity_projection_list) > 1000 * 255:
            tmp = tmp[bounding_points[0]:bounding_points[1], :]

            final_line_segment_list.append(tmp)

        if show_result:
            cv2.imshow('segmetation lines', resize_image(tmp))
            cv2.waitKey(0)

    return final_line_segment_list



"""
Word segmentation
"""


def create_hull_list(segmented_lines, show_result=True):

    result = []

    contour_list = [None] * len(segmented_lines)
    hull_list = [None] * len(segmented_lines)

    for line in segmented_lines:
        tmp = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)

        line_hull_list = []
        current_hulls = []
        current_contours = get_significant_contours(line, 50)
        contour_list.append(current_contours)
        if current_contours:
            for cnt in current_contours:
                current_hulls.append(cv2.convexHull(cnt))

        if current_hulls:
            for hull in current_hulls:
                M = cv2.moments(hull)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                line_hull_list.append([(cx, cy), hull])

        result.append(line_hull_list)

        if show_result:
            if line_hull_list:
                for hull in line_hull_list:
                    cv2.circle(tmp, hull[0], 7, (0, 0, 255), -1)
                    cv2.drawContours(tmp, [hull[1]], -1, (0, 255, 0), 2, 8)

                cv2.imshow('Convex hulls', resize_image(tmp))
                cv2.waitKey(0)

    return result


def determine_inner_hull_idices(hull_list):

    to_remove = []
    for i,j in itertools.combinations([x for x in range(len(hull_list))], 2):
        if cv2.pointPolygonTest(hull_list[j][1], hull_list[i][0], measureDist=False) > 0 and \
                cv2.contourArea(hull_list[i][1]) < cv2.contourArea(hull_list[j][1]):
            to_remove.append(i)

        elif cv2.pointPolygonTest(hull_list[i][1], hull_list[j][0], measureDist=False) > 0 and \
                cv2.contourArea(hull_list[i][1]) < cv2.contourArea(hull_list[j][1]):
            to_remove.append(j)

    to_remove = list(set(to_remove))

    return to_remove



def eliminate_inner_hulls(hull_list_of_lines, segmented_lines, show_result=True):

    for index, line in enumerate(hull_list_of_lines):

        to_remove = determine_inner_hull_idices(line)

        if show_result:
            tmp = cv2.cvtColor(segmented_lines[index], cv2.COLOR_GRAY2BGR)

            for hull_index, hull in enumerate(line):
                if hull_index in to_remove:
                    cv2.circle(tmp, hull[0], 7, (255, 0, 0), -1)
                    cv2.drawContours(tmp, [hull[1]], -1, (0, 0, 255), 3, 8)
                else:
                    cv2.circle(tmp, hull[0], 7, (255, 0, 0), -1)
                    cv2.drawContours(tmp, [hull[1]], -1, (0, 255, 0), 2, 8)

            cv2.imshow('Convex hulls', resize_image(tmp))
            cv2.waitKey(0)

        line = remove_unwanted_elements_from_list(line, to_remove)


def sort_hulls_in_lines(hull_list_of_lines):

    for line in hull_list_of_lines:
        line.sort(key = lambda x: x[0][0])


def calculate_distance_between_hulls(hull_1, hull_2):

    dist_arr = []
    for p in np.vstack(hull_2).squeeze().tolist():
        dist_arr.append(abs(cv2.pointPolygonTest(hull_1, (p[0], p[1]), measureDist=True)))

    return np.min(dist_arr)


def calculate_coherent_hull_indies(hull_list_of_lines, hull_coherency_th=60):

    # result = []
    # for line in hull_list_of_lines:
    #     indices = []
    #     if line:
    #         indices.append([0])
    #         for i in range(len(line)-1):
    #             if calculate_distance_between_hulls(line[i][1], line[i+1][1]) < hull_coherency_th:
    #                 indices[-1].append(i+1)
    #             else:
    #                 indices.append([i+1])
    #     result.append(indices)



    result = []
    for line in hull_list_of_lines:
        indices = []
        if line:
            indices.append([0])
            for i in range(1, len(line)):
                FOUND_PLACE = False
                for j in indices[-1]:
                    if calculate_distance_between_hulls(line[j][1], line[i][1]) < hull_coherency_th:
                        indices[-1].append(i)
                        FOUND_PLACE = True
                        break
                if not FOUND_PLACE:
                    indices.append([i])
        result.append(indices)

    return result


"""

"""

def determine_word_segments(source_image, segmented_lines, show_result=True):


    NUM_OF_ROWS, NUM_OF_COLS = source_image.shape

    color_list = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]

    height = 0
    for s in segmented_lines:
        height += s.shape[0]

    to_draw = np.zeros(shape=(height, NUM_OF_COLS, 3), dtype=np.uint8)
    height_index = 0

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
        print(minima_lines)

        # tmp = cv2.cvtColor(cv2.bitwise_not(segmented_lines[line_index]), cv2.COLOR_GRAY2BGR)
        tmp = cv2.cvtColor(segmented_lines[line_index], cv2.COLOR_GRAY2BGR)

        current_height = tmp.shape[0]

        closed = cv2.morphologyEx(to_morph, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))

        contours = get_significant_contours(closed, 130)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        center_points = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            center_points[i] = (
                int(boundRect[i][0]) + int(boundRect[i][2] / 2), int(boundRect[i][1]) + int(boundRect[i][3] / 2))

        boundRect = eliminate_inner_selections(boundRect)
        for box_index, box in enumerate(boundRect):

            cv2.rectangle(tmp, (int(box[0]), int(box[1])),
                          (int(box[0] + box[2]), int(box[1] + box[3])), (255, 0, 0), 5)

        to_draw[height_index:height_index + current_height, :] = tmp
        height_index += current_height

    if show_result:
        show_image(to_draw, 'Segmented words')



