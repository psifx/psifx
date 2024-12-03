def AddLLMArgument(parser):
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='name of the model, or path to a .json file containing the model specification')