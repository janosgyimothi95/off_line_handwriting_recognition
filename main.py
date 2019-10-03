from utils import *
from segmentation import *


def main():

    picture_list = sorted(os.listdir('../Pictures/Scanjob_10348/'))

    for picture in picture_list:

        # Reading img as grayscale
        test_img = cv2.imread('../Pictures/Scanjob_10348/' + picture, cv2.IMREAD_GRAYSCALE)
        print('Current picure: {}'.format(picture))

        word_dict = word_segmentation(test_img, show_results=True)


if __name__ == '__main__':

    main()