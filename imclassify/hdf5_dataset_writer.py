"""Utility class for working with hdf5 files

Modified from:
    https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/
"""
import os
import h5py


class HDF5DatasetWriter:
    """Utility for working with h5py

    Modified from:
        https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/

    :param dims: shape of data to be written to `output_path`
    :param output_path: file path to write data to
    :param data_key: name of dataset
    :param buff_size: how many rows of data to hold in memory before dumping to disk
    :param overwrite: should file at `output_path` be overwritten if it exists
    """
    def __init__(self, dims, output_path, data_key='images', buff_size=1000, overwrite=False):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path) and not overwrite:
            if not overwrite:
                raise ValueError('The supplied `output_path` already '
                                 'exists and cannot be overwritten. Manually delete '
                                 'the file before continuing.', output_path)
            else:
                os.remove(output_path)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.buff_size = buff_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        """Add data to hdf5"""
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer['data']) >= self.buff_size:
            self.flush()

    def flush(self):
        """Write the buffers to disk then reset the buffer"""
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def store_class_labels(self, class_labels):
        """Create a data set to store the actual class label names"""
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset('label_names', (len(class_labels),), dtype=dt)
        label_set[:] = class_labels

    def close(self):
        """Close connection to hdf5 file"""
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the data set
        self.db.close()
