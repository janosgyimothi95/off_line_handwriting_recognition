""" MAIN MODULE
This module's purpose is to run the program with the configured inputs.
"""


from segmentation import *
from preprocessing import *
import argparse
from argparse import RawTextHelpFormatter


def main(argument_list):
    '''Main function. Runs the progra with the configurated arguments

    :param argument_list: arguments
    :return:              -
    '''

    picture_list = argument_list[0]
    mode = argument_list[1]
    show_subresults = argument_list[2]

    if mode == 'segm':
        IMAGE_PREFIX = '../images/preprocessed/Image00'

        for i in picture_list:
            print('\nCurrent image: {}.png'.format(IMAGE_PREFIX + convert_int_to_label_string(i)))
            current_image = cv2.imread(IMAGE_PREFIX + convert_int_to_label_string(i) + '.png', cv2.IMREAD_GRAYSCALE)
            show_image(current_image, 'Preprocessed image')

            print_picture_informations(current_image)


            run_segmentation(current_image, show_subresults)

    else:
        IMAGE_PREFIX = '../images/original/Image00'

        for i in picture_list:
            print('\nCurrent image: {}.tif'.format(IMAGE_PREFIX + convert_int_to_label_string(i)))
            current_image = cv2.imread(IMAGE_PREFIX + convert_int_to_label_string(i) + '.tif')

            print_picture_informations(current_image)

            show_image(current_image, 'Original image')

            preprocessed_image = run_preprocessing(current_image, show_subresults)

            if mode == 'comb':
                run_segmentation(preprocessed_image, show_subresults)



def parse_argumments():
    ''' Parses command line arguments

    :return:    parsed argument list
    '''

    valid_id_list = [int(i.split('.')[0][-3:]) for i in sorted(os.listdir('../images/original')) if i != 'Thumbs.db']


    def valid_image_id(id):
        ''' Special type of input.

        :param id:  image id
        :return:    Exception | int id
        '''

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