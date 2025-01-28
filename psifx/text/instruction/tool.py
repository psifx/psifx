"""text instruction tool."""

from pathlib import Path
from typing import Union, Optional

from langchain_core.runnables import RunnableLambda, RunnableSerializable

from psifx.io.txt import TxtWriter, TxtReader
from psifx.io.csv import CsvWriter, CsvReader
from psifx.io.vtt import VTTReader
from psifx.text.tool import TextTool
from tqdm.auto import tqdm


class InstructionTool(TextTool):
    """
    text instruction tool.

     :param chain: The instruction chain that is applied by the tool.
     :param overwrite: Whether to overwrite existing files, otherwise raise an error.
     :param verbose: Whether to execute the computation verbosely.
     """

    def __init__(self, chain: RunnableSerializable, overwrite: Optional[bool] = False,
                 verbose: Optional[Union[bool, int]] = True):
        super().__init__(device="?", overwrite=overwrite, verbose=verbose)
        self.chain = chain

    def apply_to_txt(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        """
        Apply the instruction tool to a .txt file.

        :param input_path: Path to the input .txt file.
        :param output_path: Path to the output .txt file.
        """
        TxtWriter.check(output_path, overwrite=self.overwrite)
        text = TxtReader.read(input_path)
        result = self.chain.invoke({'text': text})

        TxtWriter.write(
            content=result,
            path=output_path,
            overwrite=self.overwrite
        )

    def apply_to_csv(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     output_column: Optional[str] = 'result'):
        """
        Apply the instruction tool to a .csv file.

        :param input_path: Path to the input .csv file.
        :param output_path: Path to the output .csv file.
        :param output_column: Name of the column to write the result of the instruction in.
        """
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

    def apply_to_vtt(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        """
        Apply the instruction tool to a .vtt file.

        :param input_path: Path to the input .vtt file.
        :param output_path: Path to the output .txt file.
        """
        TxtWriter.check(output_path, overwrite=self.overwrite)
        segments = VTTReader.read(input_path)
        text = '\n\n'.join(f"{segment['speaker']}: {segment['text']}" for segment in segments)
        result = self.chain.invoke({'text': text})

        TxtWriter.write(
            content=result,
            path=output_path,
            overwrite=self.overwrite
        )
