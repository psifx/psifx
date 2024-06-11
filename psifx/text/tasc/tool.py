import abc
import re
from pathlib import Path
from typing import Union
import pandas as pd

from psifx.io.csv import CsvWriter, CsvReader
from psifx.io.vtt import VTTReader, VTTWriter
from psifx.text.llm.tool import LLMTool


class TascTool(LLMTool, metaclass=abc.ABCMeta):

    def use(self, transcription_path, segmented_transcription_path, speaker):
        path = Path(transcription_path)
        if path.suffix == '.vtt':
            writer = TascVTTWriter
            reader = TascVTTReader
        elif path.suffix == '.csv':
            writer = CsvWriter
            reader = CsvReader
        else:
            raise NameError(path)

        writer.check(path=segmented_transcription_path,
                     overwrite=self.overwrite)
        df = reader.read(path=transcription_path)
        df = self.transform(df=df,
                            speaker=speaker)
        writer.write(df=df,
                     path=segmented_transcription_path,
                     overwrite=self.overwrite)

    @abc.abstractmethod
    def transform(self, df, speaker=None):
        pass


class TascVTTReader:

    @staticmethod
    def check(path: Union[str, Path]):
        return VTTReader.check(path=path)

    @staticmethod
    def read(
            path: Union[str, Path],
            verbose: bool = True,
    ):
        transcription = VTTReader.read(path=path)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(transcription)
        df[['text', 'segment', 'form', 'marker']] = df['text'].apply(TascVTTReader._extract_segments).apply(pd.Series)
        df = df.explode(['segment', 'form', 'marker'], ignore_index=False)
        return df

    @staticmethod
    def _extract_segments(text: str) -> tuple:
        tag_pattern = re.compile(r'<c\.segment\.*(.*?)>(.*?)</c>')
        matches = tag_pattern.findall(text)
        if matches:
            segment = [match[1] for match in matches]
            text = ' '.join(segment)
            form = [match[0].split('.')[0] if match[0] else None for match in matches]
            marker = [match[0].split('.')[1] if '.' in match[0] else None for match in matches]
        else:
            segment = None
            form = None
            marker = None
        return text, segment, form, marker


class TascVTTWriter:

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        return VTTWriter.check(path=path, overwrite=overwrite)

    @staticmethod
    def write(df: pd.DataFrame,
              path: Union[str, Path],
              overwrite: bool = False,
              verbose: bool = True, ):

        df['text'] = df.apply(TascVTTWriter._reconstruct_text, axis=1)
        print(f'''AFTER APPLY
{df}''')
        df.drop(columns=['segment', 'form', 'marker'], inplace=True, errors='ignore')

        existing_columns = [col for col in ['start', 'end', 'speaker'] if col in df.columns]
        if existing_columns:
            df.set_index(existing_columns, append=True, inplace=True)

        df = df.groupby(level=list(range(df.index.nlevels)))['text'].apply(' '.join).reset_index()
        print(f'''AFTER GROUPBY 
{df}''')
        VTTWriter.write(
            segments=df.to_dict(orient='records'),
            path=path,
            overwrite=overwrite,
            verbose=verbose,
        )

    @staticmethod
    def _reconstruct_text(row) -> str:
        if row['segment']:
            return f"<c.segment{'.' + row.get('form') if row.get('form') else ''}{'.' + row.get('marker') if row.get('marker') else ''}> {row['segment']} </c>"
        else:
            return row['text']
