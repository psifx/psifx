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
        self._set_instruction(instruction)
        self._set_parser(start_flag, separator)

    def _set_instruction(self, instruction):
        try:
            instruction = TxtReader.read(path=instruction)
        except NameError:
            pass

        self.instruction = ChatPromptTemplate.from_template(template=instruction)
        # lambda text: ChatPromptTemplate.from_template(template=instruction).invoke(text) # same as .invoke({'one_field', text})

    def _set_parser(self, start_flag: str, separator: str):

        def parser(generation: AIMessage, message: str):
            answer = generation.content.split(start_flag)[-1]
            segments = answer.split(separator)
            segments = [segment.strip() for segment in segments]

            reconstruction = []
            remaining_message = message

            for segment in segments[:-1]:
                if segment:
                    match = re.search(re.escape(segment), remaining_message)
                    if match:
                        reconstruction.append(match.group().strip())
                        remaining_message = remaining_message[match.end():]

            if remaining_message.strip():
                reconstruction.append(remaining_message.strip())

            if reconstruction != segments:
                print(
                    f"PROBLEMATIC GENERATION: {generation.content}\nINPUT: {message}\nRECONSTRUCTION: {reconstruction}")
            return reconstruction

        self.parser = parser

    @staticmethod
    def _to_string_css(messages: list[str]):
        return ' '.join([f"<c.segment>{s}</c>" for s in messages])

    @staticmethod
    def _dict_wrapper(function):
        return lambda dictionary: function(**dictionary)

    def get_chain(self) -> Chain:
        return RunnableParallel(
            {'generation': self.instruction | self.llm, 'message': RunnablePassthrough()}) | self._dict_wrapper(
            self.parser)

    def segment_vtt(self, transcription_path, segmented_transcription_path, speaker):
        VTTWriter.check(segmented_transcription_path, overwrite=self.overwrite)
        transcription = VTTReader.read(path=transcription_path)
        df = pd.DataFrame(transcription)
        if not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the .vtt file")

        chain = self.get_chain() | self._to_string_css

        df.loc[df['speaker'] == speaker, 'text'] = df.loc[df['speaker'] == speaker, 'text'].apply(
            lambda message: chain.invoke(message))

        print(df.to_dict(orient='records'))
        VTTWriter.write(
            segments=df.to_dict(orient='records'),
            path=segmented_transcription_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

    def segment_csv(self, transcription_path, segmented_transcription_path, speaker):
        CsvWriter.check(segmented_transcription_path, overwrite=self.overwrite)
        df = CsvReader.read(path=transcription_path)
        if speaker not in df.columns:
            raise ValueError(f"The column {speaker} is not found in the .csv file")

        chain = self.get_chain()

        df['segment'] = df[speaker].apply(lambda message: chain.invoke(message))
        df = df.explode('segment').reset_index(drop=True)
        CsvWriter.write(
            df=df,
            path=segmented_transcription_path,
            overwrite=self.overwrite,
        )

    def segment(self, transcription_path, segmented_transcription_path, speaker):
        try:
            return self.segment_vtt(transcription_path, segmented_transcription_path, speaker)
        except NameError:
            try:
                return self.segment_csv(transcription_path, segmented_transcription_path, speaker)
            except NameError:
                print(
                    f"Transcription {transcription_path} and segmentation {segmented_transcription_path} should both be .vtt files or both be .csv files")
