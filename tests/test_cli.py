from data_annotation_agent.cli import build_parser


def test_cli_defaults_to_pixel_coordinate_system() -> None:
    parser = build_parser()
    args = parser.parse_args(["annotate", "--input-dir", "C:/tmp/images", "--labels", "poppy"])

    assert args.coordinate_system == "pixel"
    assert args.max_edge == 1024
    assert args.provider is None


def test_cli_accepts_provider_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["annotate", "--input-dir", "C:/tmp/images", "--labels", "poppy", "--provider", "ollama"]
    )

    assert args.provider == "ollama"
