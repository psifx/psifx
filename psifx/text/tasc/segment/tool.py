from typing import Union

from langchain_core.runnables import RunnableLambda
from psifx.text.tasc.tool import TascTool


class SegmentTool(TascTool):
    """
    Base class for Segmentation.
    """

    def __init__(self, model, instruction: str, parser: dict, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(model, overwrite, verbose)
        self.instruction = self.load_template(template=instruction)
        self.parser = self.make_parser(**parser, verbose=verbose)
        self.chain = (RunnableLambda(lambda x: x.to_dict()) | self.get_chain(instruction=self.instruction, parser=self.parser))

    def transform(self, df, speaker=None):
        if speaker and not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the file")
        condition = df['speaker'] == speaker if speaker else slice(None)
        if 'segment' in df.columns and df.loc[condition, 'segment'].any():
            print("WARNING already segmented")
        df.loc[condition, 'segment'] = df.loc[condition, ['text']].apply(self.chain.invoke, axis=1)
        df = df.explode(['segment'], ignore_index=False)
        return df
