import re
from typing import Union
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from psifx.text.llm.tool import LLMTool
from psifx.text.tasc.tool import TascTool


class MarkerTool(TascTool):
    """
    Base class for Marker.
    """

    def __init__(self, model, instructions: dict, start_flag: str, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(model, overwrite, verbose)
        self.instructions = {k: self.load_template(v) for k, v in instructions.items()}
        self.parser = lambda generation: MarkerTool.default_parser(generation=generation,
                                                                   start_flag=start_flag,
                                                                   expected_labels=None)
        self.chain = self.instructions | self.llm | self.parser
        # behavior differ according to form,
        # 

    def transform(self, df, speaker=None):
        if speaker and not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the file")
        condition = df['speaker'] == speaker if speaker else slice(None)
        # df.loc[condition, 'marker'] = df.loc[condition, 'form'].apply(self.chain.invoke)
        raise NotImplementedError
