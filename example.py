from imclassify import ImClassifier

im_classifier = ImClassifier(labels=['no_fork', 'fork'])
im_classifier.gather_images(video_path=0)
im_classifier.extract_features()
im_classifier.train_model(percent_train=1.0)
im_classifier.classify_video()
