from typing import Union

from langchain_core.runnables import RunnableLambda

from psifx.io.csv import CsvWriter, CsvReader
from psifx.text.llm.tool import LLMUtility
from psifx.tool import Tool


class InstructionTool(Tool):
    """
    Base class for Custom instruction, take a .csv for input and .yaml file for prompt template
    """

    def __init__(self, chain, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(overwrite, verbose)
        self.chain = RunnableLambda(lambda x: x.to_dict()) | chain

    def apply_to_csv(self, input_path, output_path, output_column='result'):
        CsvWriter.check(output_path, overwrite=self.overwrite)
        df = CsvReader.read(path=input_path)
        df[output_column] = df.apply(self.chain.invoke, axis=1)
        CsvWriter.write(
            df=df,
            path=output_path,
            overwrite=self.overwrite,
        )
