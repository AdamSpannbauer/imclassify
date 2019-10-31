import argparse
from gooey import Gooey
from imclassify import ImClassifier


def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x


@Gooey(program_name='imclassify')
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--labels', default='class_1 class_2 class_3', nargs='+',
                    help='List of class labels to be used (separated by space).')
    ap.add_argument('-t', '--train_input', type=try_int, default='0',
                    help='Path to input video to capture training data from '
                         '(can be number to indicate webcam; see cv2.VideoCapture() docs).')
    ap.add_argument('-o', '--train_output', default='images',
                    help='Main dir for training images to be saved to '
                         '(they will be saved to a subdir named by the class label).')
    ap.add_argument('-d', '--feature_db', default='features.hdf5',
                    help='Path to save HDF5 file of features to.')
    ap.add_argument('-m', '--model_output', default='model.pickle',
                    help='Path to save pickled sklearn model to.')
    args = vars(ap.parse_args())

    im_classifier = ImClassifier(labels=args['labels'],
                                 images_path=args['train_output'],
                                 features_path=args['feature_db'],
                                 model_path=args['model_output'])

    im_classifier.gather_images(video_path=args['train_input'])
    im_classifier.extract_features()
    im_classifier.train_model(percent_train=1.0)
    im_classifier.classify_video(video_path=args['train_input'], output_path='test_output.avi')


if __name__ == '__main__':
    main()
