from typing import Union

import pandas as pd
from langchain_core.runnables import RunnableLambda

from psifx.io.csv import CsvWriter, CsvReader
from psifx.text.llm.tool import LLMTool


class InstructionTool(LLMTool):
    """
    Base class for Custom instruction, take a .csv for input and .yaml file for prompt template
    """

    def __init__(self, model, instruction: str, parser: dict, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(model, overwrite, verbose)
        self.instruction = self.load_template(template=instruction)
        self.parser = self.make_parser(**parser)
        self.chain = (RunnableLambda(lambda x: x.to_dict()) | self.get_chain(instruction=self.instruction, parser=self.parser))

    def segment_csv(self, input_path, output_path, output_column='result'):
        CsvWriter.check(output_path, overwrite=self.overwrite)
        df = CsvReader.read(path=input_path)
        df[output_column] = df[self.instruction.input_variables].apply(self.chain.invoke, axis=1)
        CsvWriter.write(
            df=df,
            path=output_path,
            overwrite=self.overwrite,
        )
