"""Train logistic regression model on hdf5 features for classification

Modified from:
    https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/
"""
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_model(h5py_db, model_output='model.pickle', percent_train=1.0):
    """Train logistic regression classifier

    :param h5py_db: path to HDF5 database containing 'features', 'labels', & 'label_names'
    :param model_output: path to save trained model to using pickle
    :param percent_train: percent of images to be used for training (instead of testing)
    :return: None; output is written to `model_output`
    """

    i = int(h5py_db['labels'].shape[0] * percent_train)

    # C decided with sklearn.model_selection.GridSearchCV
    model = LogisticRegression(C=0.1)
    model.fit(h5py_db['features'][:i], h5py_db['labels'][:i])

    if percent_train < 1.0:
        preds = model.predict(h5py_db['features'][i:])
        print(classification_report(h5py_db['labels'][i:], preds,
                                    target_names=h5py_db['label_names']))

    with open(model_output, 'wb') as f:
        f.write(pickle.dumps(model))
