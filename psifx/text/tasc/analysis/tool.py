import re
from itertools import zip_longest, chain, accumulate
from typing import Union

from psifx.io.csv import CsvReader
from psifx.tool import Tool
from psifx.io.txt import TxtWriter
from psifx.io.vtt import VTTReader
import pandas as pd


class AnalysisTool(Tool):
    """
    Base class for TASc Analysis.
    """

    @staticmethod
    def compare(prediction, truth):
        """
        Take as input two lists containing parts of one sentence
        :return: tp fp and fn
        """

        # Split words in each parts
        prediction, truth = map(lambda lst: [part.split() for part in lst], (prediction, truth))

        # Check that the splitted words match between the two lists, gives warning otherwise
        for i, (word1, word2) in enumerate(zip_longest(chain(*prediction), chain(*truth), fillvalue=None)):
            if word1 != word2:
                print(f"Warning: Mismatch at position {i}: '{word1}' != '{word2}'")

        # Compute position of separators in terms of words count
        prediction, truth = map(lambda split: set(accumulate(len(part) for part in split[:-1])), (prediction, truth))

        # Compute tp, fp and fn for matching positions while allowing a +-1 error
        tp = sum(1 for a in truth if a in prediction or a + 1 in prediction or a - 1 in prediction)  # True positives
        fp = len(prediction) - tp  # False positives
        fn = len(truth) - tp  # False negatives
        return tp, fp, fn

    @staticmethod
    def f1_score(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def evaluation(self, prediction_path, truth_path, speaker, result_path):
        TxtWriter.check(result_path, overwrite=self.overwrite)
        df_pred = CsvReader.read(prediction_path)
        df_truth = CsvReader.read(truth_path)

        results = [self.compare(prediction=prediction, truth=truth) for prediction, truth in
                   zip(df_pred['text'], df_truth['text'])]
        total_tp, total_fp, total_fn = map(sum, zip(*results))
        precision, recall, f1 = self.f1_score(tp=total_tp, fp=total_fp, fn=total_fn)
        message = f"""Precision {precision}
Recall {recall}
F1 {f1}
"""
        TxtWriter.write(message, result_path)
