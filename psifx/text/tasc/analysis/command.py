import argparse

from psifx.utils.command import Command
from psifx.text.tasc.analysis.tool import AnalysisTool


class AnalysisCommand(Command):
    """
    Tool for TASc Evaluation
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )
        parser.add_argument(
            '--truth',
            type=str,
            required=True,
            help="path to the true segmentation .vtt or .csv file")
        parser.add_argument(
            '--prediction',
            type=str,
            required=True,
            help="path to the predicted segmentation .vtt or .csv file")
        parser.add_argument(
            '--speaker',
            type=str,
            default=None,
            help="name of the speaker for .vtt or .csv transcription")
        parser.add_argument(
            '--result',
            type=str,
            required=True,
            help="path to the output result .txt file")

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        AnalysisTool(
            overwrite=args.overwrite,
            verbose=args.verbose
        ).evaluation(
            prediction_path=args.prediction,
            truth_path=args.truth,
            speaker=args.speaker,
            result_path=args.result
        )
