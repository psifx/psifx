def AddLLMArgument(parser):
    parser.add_argument(
        '--llm',
        type=str,
        required=True,
        help='path to a .yaml file containing the large language model specifications')