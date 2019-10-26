import os
import glob
import string
import h5py
from .gather_images import prompt_labels, gather_images
from .extract_features import extract_features
from .train_model import train_model
from .classifier import Classifier, video_classify


class ImClassifier:
    def __init__(self, labels=None, images_path='images', features_path='features.hdf5', model_path='model.pickle'):
        self.labels, self._label_keys = self._read_labels(labels)

        self.images_path = images_path
        self.features_path = features_path
        self.model_path = model_path
        self._classifier = None

    @staticmethod
    def _read_labels(labels):
        if labels is None:
            label_keys = prompt_labels()
            labels = list(label_keys.values())
        else:
            if len(labels) > 26:
                raise ValueError('Only supports up to 26 classes.')

            keys = list(string.ascii_lowercase[:len(labels)])
            label_keys = {k: v for k, v in zip(keys, labels)}

        return labels, label_keys

    def gather_images(self, video_path=0):
        """Collect training images from video

        :param video_path: Path to input video or camera index (see: `cv2.VideoCapture()`)
        :return: None; output saved to dir specified by `images_path` attribute
        """
        gather_images(output_dir=self.images_path,
                      labels=self.labels,
                      video_capture=video_path)

    def extract_features(self, batch_size=32, buffer_size=1000):
        """Extract features from images at images_path and write to features_path

        :param batch_size: Number of images to process at a time
        :param buffer_size: Memory threshold for features before writing to disk
        :return: None
        """
        image_paths = []
        for label in self.labels:
            glob_path = os.path.join(self.images_path, label, '*')
            image_paths.extend(glob.glob(glob_path))

        extract_features(image_paths,
                         hdf5_file=self.features_path,
                         batch_size=batch_size,
                         buffer_size=buffer_size)

    def train_model(self, percent_train=1.0):
        """Train model using the provided labels and features at features_path

        :param percent_train: percent of images to be used for training (instead of testing)
        :return: None
        """
        with h5py.File(self.features_path, 'r') as db:
            self.labels = list(db['label_names'])
            train_model(db, self.model_path, percent_train=percent_train)

    def _init_classifier(self):
        if self._classifier is None:
            self._classifier = Classifier(labels=self.labels,
                                          model_path=self.model_path)

    def classify_image(self, image):
        """Classify a single image

        :param image: image to classify
        :return: tuple of label, probabilities; where label is the predicted class
                 and probabilities is a list of showing the probability for each
                 class
        """
        self._init_classifier()
        return self._classifier.predict(image)

    def classify_video(self, video_path=0, output_path=None):
        """Demo classification on video

        :param video_path: Path to input video or camera index (see: `cv2.VideoCapture()`)
        :param output_path: Optional output path for session to be recorded to
        :return: None
        """
        self._init_classifier()
        video_classify(classifier=self._classifier,
                       video_capture=video_path,
                       output_path=output_path)
