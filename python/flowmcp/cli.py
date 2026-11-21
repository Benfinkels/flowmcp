import argparse
import sys
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowmcp",
        description="Run flowmcp pipelines from the terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using an LLM and optional documents.",
    )
    ask_parser.add_argument(
        "--llm",
        required=True,
        help="LLM provider/model string ^(e.g., openai/gpt-4^).",
    )
    ask_parser.add_argument(
        "--docs",
        default=None,
        help="Path to document file or directory.",
    )
    ask_parser.add_argument(
        "question",
        help="Question to ask the pipeline.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        pipeline = Pipeline(llm=args.llm, docs=args.docs)
        answer = pipeline.ask(args.question)
        print(answer)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
