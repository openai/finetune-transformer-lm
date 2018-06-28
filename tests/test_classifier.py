import os
import unittest
import logging
from copy import copy
from pathlib import Path

# required for tensorflow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import enso
from enso.download import generic_download
from sklearn.metrics import accuracy_score

from finetune import LanguageModelClassifier

SST_FILENAME = "SST-binary.csv"


class TestLanguageModelClassifier(unittest.TestCase):

    n_sample = 100
    n_hidden = 768
    dataset_path = os.path.join(
        enso.config.DATA_DIRECTORY, 'Classify', 'SST-binary.csv'
    )
    @classmethod
    def _download_sst(cls):
        """
        Download Stanford Sentiment Treebank to enso `data` directory
        """
        path = Path(cls.dataset_path)
        if path.exists():
            return

        path.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )


    @classmethod
    def setUpClass(cls):
        cls._download_sst()

    def setUp(self):
        self.dataset = pd.read_csv(self.dataset_path)
        tf.reset_default_graph()

    def test_fit_predict(self):
        """
        Ensure model training does not error out
        Ensure model returns predictions of the right type
        """
        save_file_autosave = 'tests/saved-models/autosave_path'
        model = LanguageModelClassifier(verbose=False, autosave_path=save_file_autosave)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)

        predictions = model.predict(valid_sample.Text)
        for prediction in predictions:
            self.assertIsInstance(prediction, (np.int, np.int64))

        probabilities = model.predict_proba(valid_sample.Text)
        for proba in probabilities:
            self.assertIsInstance(proba, dict)

    def test_save_load(self):
        """
        Ensure saving + loading does not cause errors
        Ensure saving + loading does not change predictions
        """
        save_file_autosave = 'tests/saved-models/autosave_path'
        save_file = 'tests/saved-models/test-save-load'
        model = LanguageModelClassifier(verbose=False, autosave_path=save_file_autosave)
        train_sample = self.dataset.sample(n=self.n_sample)
        valid_sample = self.dataset.sample(n=self.n_sample)
        model.fit(train_sample.Text, train_sample.Target)
        predictions = model.predict(valid_sample.Text)
        model.save(save_file)
        model = LanguageModelClassifier.load(save_file)
        new_predictions = model.predict(valid_sample.Text)
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction, new_predictions[i])

    def test_featurize(self):
        """
        Ensure featurization returns an array of the right shape
        Ensure featurization is still possible after fit
        """
        save_file_autosave = 'tests/saved-models/autosave_path'
        model = LanguageModelClassifier(verbose=False, autosave_path=save_file_autosave)
        train_sample = self.dataset.sample(n=self.n_sample)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, self.n_hidden))
        model.fit(train_sample.Text, train_sample.Target)
        features = model.featurize(train_sample.Text)
        self.assertEqual(features.shape, (self.n_sample, self.n_hidden))

    def test_reasonable_predictions(self):
        save_file_autosave = 'tests/saved-models/autosave_path'
        model = LanguageModelClassifier(verbose=False, autosave_path=save_file_autosave)
        trX = ['cat'] * 100 + ['finance']  * 100
        trY = copy(trX)
        teX = ['feline'] * 10 + ['investment'] * 10
        teY = ['cat'] * 10 + ['finance'] * 10
        model.fit(trX, trY)
        predY = model.predict(teX)
        self.assertEqual(accuracy_score(teY, predY), 1.00)
