from typing import Union

from langchain_core.runnables import RunnableLambda

from psifx.text.tasc.tool import TascTool

import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize


class SegmentTool(TascTool):
    """
    Base class for Segmentation.
    """

    def __init__(self, find_separators, make_sense, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(device="?", overwrite=overwrite, verbose=verbose)
        try:
            find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.find_separators = find_separators
        self.make_sense = make_sense

    def transform(self, df, speaker=None):
        if speaker and not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the file")
        condition = df['speaker'] == speaker if speaker else slice(None)

        if 'segment' in df.columns and df.loc[condition, 'segment'].any():
            print("WARNING already segmented")
        df.loc[condition, 'segment'] = df.loc[condition, ['text']].apply(self.iterative_process, axis=1)

        df = df.explode(['segment'], ignore_index=False)
        return df

    def iterative_process(self, text):
        print("ITERATIVE PROCESS")
        text = text['text']
        print(text)
        basic_segmentation = sent_tokenize(text)
        print(basic_segmentation)
        segmentations = [self.find_separators.invoke({'text': part}) for part in basic_segmentation]
        segments = [""]
        for segmentation in segmentations:
            segments[-1] = (segments[-1] + " " + segmentation[0]).strip()
            segments += segmentation[1:]
        segments = [segment for segment in segments if segment]
        print(segments)
        sentences, current_segment = [], ""

        for seg in segments:
            current_segment = f"{current_segment} {seg}".strip()

            if self.make_sense.invoke(current_segment) == "yes":
                sentences.append(current_segment)
                current_segment = ""

        if current_segment:
            sentences.append(current_segment)

        print(sentences)
        return sentences
