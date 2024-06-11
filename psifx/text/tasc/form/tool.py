from typing import Union

from psifx.text.tasc.tool import TascTool


class FormTool(TascTool):
    """
    Base class for Segmentation.
    """

    def __init__(self, model, instruction: str, start_flag: str, separator: str, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(model, overwrite, verbose)
        self.instruction = self.load_template(instruction)
        self.parser = lambda generation, message: FormTool.default_parser(generation=generation,
                                                                          message=message,
                                                                          start_flag=start_flag,
                                                                          expected_labels=["inquiry", "expression", "action",
                                                                                   "education", "clarification",
                                                                                   "conjecture", "assertion"])
        self.chain = self.instruction | self.llm | self.parser

    def transform(self, df, speaker=None):
        if speaker and not (df['speaker'] == speaker).any():
            raise ValueError(f"The speaker {speaker} is not found in the file")
        condition = df['speaker'] == speaker if speaker else slice(None)
        df.loc[condition, 'form'] = df.loc[condition, 'segment'].apply(self.chain.invoke)
