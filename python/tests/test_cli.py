from unittest.mock import MagicMock, patch
from flowmcp.cli import main


def test_cli_ask_with_docs(capsys):
    mock_instance = MagicMock()
    mock_instance.ask.return_value = "30-day full refund."

    with patch("flowmcp.cli.Pipeline", return_value=mock_instance) as mock_pipeline_cls:
        exit_code = main([
            "ask",
            "--llm", "openai/gpt-4",
            "--docs", "./my-docs",
            "What is our refund policy?",
        ])

    assert exit_code == 0
    mock_pipeline_cls.assert_called_once_with(llm="openai/gpt-4", docs="./my-docs")
    mock_instance.ask.assert_called_once_with("What is our refund policy?")
    captured = capsys.readouterr()
    assert "30-day full refund." in captured.out
