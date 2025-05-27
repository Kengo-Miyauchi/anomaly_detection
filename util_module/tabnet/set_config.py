import argparse
def set_config_file():
    # 引数を定義
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    # 引数を解析
    args = parser.parse_args()
    return args.config