

def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x


if __name__ == '__main__':
    import argparse
    from imclassify import ImClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--train_input', type=try_int, default=0,
                    help='Path to input video to capture training data from. '
                         '(can be number to indicate web cam; see cv2.VideoCapture docs)')
    ap.add_argument('-l', '--labels', default=None, nargs='+',
                    help='List of class labels to be used (separated by space)')
    ap.add_argument('-o', '--train_output', default='images',
                    help='Main dir for training images to be saved to. '
                         '(they will saved to a subdir defined by FLAGS dict)')
    ap.add_argument('-d', '--feature_db', default='features.hdf5',
                    help='path to output HDF5 file')
    ap.add_argument('-m', '--model_output', default='model.pickle',
                    help='path to output model to')
    args = vars(ap.parse_args())

    im_classifier = ImClassifier(labels=args['labels'])
    im_classifier.gather_images(video_path=0)
    im_classifier.extract_features()
    im_classifier.train_model(percent_train=1.0)
    im_classifier.classify_video()
