""" MAIN MODULE

"""


from segmentation import *
from preprocessing import *
import argparse
from argparse import RawTextHelpFormatter


def main(argument_list):
    '''

    :return:
    '''

    picture_list = argument_list[0]
    mode = argument_list[1]
    show_subresults = argument_list[2]

    if mode == 'segm':
        IMAGE_PREFIX = '../../Images/preprocessed/Image00'

        for i in picture_list:
            current_image = cv2.imread(IMAGE_PREFIX + convert_int_to_label_string(i) + '.png', cv2.IMREAD_GRAYSCALE)
            show_image(current_image, 'Preprocessed image')

            # TODO Image description

            run_segmentation(current_image, show_subresults)

    else:
        IMAGE_PREFIX = '../../Images/magyar/Image00'

        for i in picture_list:
            current_image = cv2.imread(IMAGE_PREFIX + convert_int_to_label_string(i) + '.tif')
            # TODO Image description

            show_image(current_image, 'Original image')

            preprocessed_image = run_preprocessing(current_image, show_subresults)

            if mode == 'comb':
                run_segmentation(preprocessed_image, show_subresults)



def parse_argumments():

    valid_id_list = [int(i.split('.')[0][-3:]) for i in sorted(os.listdir('../../Images/magyar')) if i != 'Thumbs.db']


    def valid_image_id(id):
        id = int(id)
        if id not in valid_id_list:
            raise argparse.ArgumentTypeError("\nSpecified image id: {} was not found in image list.".format(id))
        return id


    parser = argparse.ArgumentParser(description='Offline handwriting recognition system:\nPreprocessing & Segmentation modules', formatter_class = RawTextHelpFormatter)

    parser.add_argument('-i', '--images', nargs='+', default=valid_id_list, metavar="", type=valid_image_id, help='Specify which images to use. \nValid image ids: {}\n\n'.format(valid_id_list))

    parser.add_argument('-m', '--mode',  choices=['prep', 'segm', 'comb'], default='comb', help='Specifies the running mode of the program\n\n')

    parser.add_argument('-d', '--detailed', action='store_true', help='Sets the display to be detailed, meaning the program will show subresults')


    args = parser.parse_args()

    args.images = list(set(args.images))
    args.images.sort()

    argument_list = [args.images, args.mode, args.detailed]

    return argument_list



if __name__ == '__main__':
    main(parse_argumments())