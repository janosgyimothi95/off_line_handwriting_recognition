from utils import *
from segmentation import *
from preprocessing import *

def main():

    # picture_list = sorted(os.listdir('../../Pictures/Scanjob_10348/'))
    # print(picture_list)
    # for picture in picture_list:
    #
    #     # Reading img as grayscale
    #     test_img = cv2.imread('../../Pictures/Scanjob_10343/' + picture, cv2.IMREAD_GRAYSCALE)
    #     print('Current picure: {}'.format(picture))
    #
    #     word_dict = word_segmentation(test_img, show_results=True)


    CRITICAL_PICTURES = ['071', '113', '114']
    CRITICAL_BAD_ROIS = ['014', '035', '072', '027', '108']
    BAD_ROIS = ['016','017','018','035','042', '047', '051', '072', '083', '111']
    TEST = ['082']
    GOOD_ROTATION = ['012']
    UNDERLINE_TESTCASES = ['001', '010', '019', '022', '023', '027', '032', '034', '045', '046', '050', '051', '053',
                           '054', '056', '061', '062', '063', '064', '066', '081', '082', '085', '091', '092', '095',
                           '105', '106']

    UNDERLINE_CRITICAL = ['024', '056', '063', '065', '026', '028', '029', '099']
    UNDERLINE_TEST = ['085']



    picture_list = sorted(os.listdir('../../Images/magyar/'))

    print(picture_list)



    for picture_label in UNDERLINE_CRITICAL:
        rusted_picture = cv2.imread('../../Images/magyar/Image00' + picture_label + '.tif')
        # rusted_picture = cv2.imread('../../Images/magyar/' + picture_label)


        print('\nCurrent picture: ', picture_label.split('.')[0][-3:])


        #RUST REMOVAL
        rust_mask = rust_detection(rusted_picture, False)
        smooth_mask = smooth_rust_mask(rust_mask, False)
        rustless = eliminate_rust_points(rusted_picture, smooth_mask, False)


        #BINARIZATION
        gray_rustless = convert_image_to_grayscale(rustless, False)
        th = determine_binary_threshold(gray_rustless, False)
        binary_rustless = binarize_grayscale_image(gray_rustless, th, False)


        #DOCUMENT IDENTIFIER REMOVAL
        mask, boxes = remove_document_identifiers(binary_rustless, False)
        masked_image = cv2.bitwise_and(mask, binary_rustless)


        #ROI SELECTION AND CROPPING
        roi = select_roi(cv2.bitwise_and(mask, binary_rustless), False)
        cropped = crop_roi(masked_image, roi, False)


        #ROI ROTATION
        rotated = rotate_image(cropped, False)


        #UNDERLINE ELIMINATION
        underlines = detect_underlines(rotated, False)
        _ = mask_detected_underlines(rotated, underlines, True)





if __name__ == '__main__':

    main()