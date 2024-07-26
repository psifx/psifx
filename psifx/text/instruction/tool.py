from typing import Union

from langchain_core.runnables import RunnableLambda

from psifx.io.txt import TxtWriter, TxtReader
from psifx.io.csv import CsvWriter, CsvReader
from psifx.tool import Tool
from tqdm.auto import tqdm


class InstructionTool(Tool):
    """
    Base class for Custom instruction
    """

    def __init__(self, chain, overwrite: bool = False,
                 verbose: Union[bool, int] = True):
        super().__init__(device="?", overwrite=overwrite, verbose=verbose)
        self.chain = chain

    def apply_to_txt(self, input_path, output_path, output_column='result'):
        TxtWriter.check(output_path, overwrite=self.overwrite)
        text = TxtReader.read(input_path)
        result = self.chain.invoke({'text': text})

        TxtWriter.write(
            content=result,
            path=output_path,
            overwrite=self.overwrite
        )

    def apply_to_csv(self, input_path, output_path, output_column='result'):
        CsvWriter.check(output_path, overwrite=self.overwrite)
        df = CsvReader.read(path=input_path)
        df_chain = RunnableLambda(lambda x: x.to_dict()) | self.chain
        tqdm.pandas(desc="Processing")
        df[output_column] = df.progress_apply(df_chain.invoke, axis=1)
        CsvWriter.write(
            df=df,
            path=output_path,
            overwrite=self.overwrite,
        )
