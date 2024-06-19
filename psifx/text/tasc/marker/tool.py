import re
from typing import Union
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from psifx.text.llm.tool import LLMUtility
from psifx.text.tasc.tool import TascTool


class MarkerTool(TascTool):
    """
    Base class for Marker.
    """

    def __init__(self, chains, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(overwrite, verbose)
        self.chains = chains
        # behavior differ according to form,

    def transform(self, df, speaker=None):
        if speaker and not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the file")
        condition = df['speaker'] == speaker if speaker else slice(None)
        # df.loc[condition, 'marker'] = df.loc[condition, 'form'].apply(self.chain.invoke)
        raise NotImplementedError
