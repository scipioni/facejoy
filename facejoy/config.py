import argparse

import configargparse

_config = None


def get_config() -> argparse.Namespace:
    global _config

    if _config is not None:
        return _config

    parser = configargparse.get_argument_parser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", required=False, is_config_file=True, help="config file path"
    )

    parser.add_argument(
        "--config-save",
        required=False,
        is_write_out_config_file_arg=True,
        help="config file path",
    )

    # parser.add_argument("files", nargs="*", default=[], help="list of documents")
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    parser.add_argument(
        "--position-samples",
        default=25,
        type=int,
        help="Number of samples to smooth the position",
    )
    parser.add_argument(
        "--mouth-open-threshold",
        default=.4,
        type=float,
        help="Mouth open threshold for click detection",
    )
    _config, _ = parser.parse_known_args()

    return _config


config = get_config()
