import re
from typing import Union

from langchain.chains.base import Chain
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from psifx.io.csv import CsvWriter, CsvReader
from psifx.text.llm.tool import LLMTool
from langchain_core.prompts import ChatPromptTemplate
from psifx.io.txt import TxtReader
from psifx.io.vtt import VTTReader, VTTWriter
import pandas as pd


class TascTool(LLMTool):
    """
    Base class for TASc.
    """

    def __init__(self, model, instruction: str, start_flag: str, separator: str, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(model, overwrite, verbose)
        self.instruction = self.load_template(instruction)
        self.parser = lambda generation, message: self.split_parser(generation=generation,
                                                                    message=message,
                                                                    start_flag=start_flag,
                                                                    separator=separator)
        self.chain = RunnableParallel(
            {'generation': self.instruction | self.llm, 'message': RunnablePassthrough()}) | self._dict_wrapper(
            self.parser)

    def read_vtt(self, transcription_path, speaker):
        transcription = VTTReader.read(path=transcription_path)
        df = pd.DataFrame(transcription)
        if not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the .vtt file")
        df.loc[df['speaker'] == speaker, 'text'].apply(self.from_string_css)
        return df

    def segment(self, df, speaker=None):
        condition = df['speaker'] == speaker if speaker else slice(None)
        df.loc[condition, 'text'] = df.loc[condition, 'text'].apply(self.chain.invoke)  # Just self.chain ?

    def save_vtt(self, df, speaker, segmented_transcription_path):
        df.loc[df['speaker'] == speaker, 'text'] = df.loc[df['speaker'] == speaker, 'text'].apply(self._to_string_css)
        VTTWriter.write(
            segments=df.to_dict(orient='records'),
            path=segmented_transcription_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

    @staticmethod
    def _to_string_css(messages: list[str]):
        return ' '.join([f"<c.segment>{s}</c>" for s in messages])

    @staticmethod
    def from_string_css(css_string: str) -> list[str]:
        # Use a regular expression to find all segments
        pattern = r"<c\.segment>(.*?)<\/c>"
        segments = re.findall(pattern, css_string)
        return segments

    def segment_vtt(self, transcription_path, segmented_transcription_path, speaker):
        VTTWriter.check(segmented_transcription_path, overwrite=self.overwrite)
        df = self.read_vtt(transcription_path=transcription_path, speaker=speaker)
        self.segment(df=df, speaker=speaker)
        self.save_vtt(df=df, speaker=speaker, segmented_transcription_path=segmented_transcription_path)

    def segment_csv(self, transcription_path, segmented_transcription_path, speaker):
        CsvWriter.check(segmented_transcription_path, overwrite=self.overwrite)
        df = CsvReader.read(path=transcription_path)
        self.segment(df=df, speaker=speaker)
        CsvWriter.write(
            df=df,
            path=segmented_transcription_path,
            overwrite=self.overwrite,
        )

    def segment(self, transcription_path, segmented_transcription_path, speaker):
        output = None
        try:
            VTTWriter.check(segmented_transcription_path, overwrite=self.overwrite)
            output = '.vtt'
        except NameError:
            try:
                CsvWriter.check(segmented_transcription_path, overwrite=self.overwrite)
                output = '.csv'
            except NameError:
                print(
                    f"The output segmentation path {segmented_transcription_path} should be a .vtt or a .csv file.")
                return
        try:
            df = self.read_vtt(transcription_path=transcription_path, speaker=speaker)
        except NameError:
            try:
                df = CsvReader.read(path=transcription_path)
            except NameError:
                print(
                    f"The transcription path {transcription_path} should be a .vtt or a .csv file.")
                return
        self.segment(df=df, speaker=speaker)
        if output == '.vtt':
            self.save_vtt(df=df, speaker=speaker, segmented_transcription_path=segmented_transcription_path)
        elif output == '.csv':
            CsvWriter.write(
                df=df,
                path=segmented_transcription_path,
                overwrite=self.overwrite,
            )
