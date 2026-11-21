
import re
import sys
import types
from unittest.mock import MagicMock
from flowmcp.rag.loader import RAG


class _FakeSoup:
    def __init__(self, html: str, parser: str):
        self._html = html

    def __call__(self, tags):
        return []

    def get_text(self, separator: str = " ", strip: bool = False) -> str:
        cleaned = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            "",
            self._html,
            flags=re.S | re.I,
        )
        text = re.sub(r"<[^>]+>", separator, cleaned)
        parts = [p.strip() for p in text.split() if p.strip()]
        return separator.join(parts) if strip else " ".join(parts)


def test_rag_loads_and_chunks_url(monkeypatch):
    html = (
        "<html><head><style>body{}</style></head>"
        "<body><h1>FAQ</h1><p>30-day full refund.</p><script>bad()</script></body></html>"
    )
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status.return_value = None

    mock_get = MagicMock(return_value=mock_response)
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.get = mock_get

    fake_bs4 = types.ModuleType("bs4")
    fake_bs4.BeautifulSoup = _FakeSoup

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "bs4", fake_bs4)

    mock_retriever = MagicMock()
    monkeypatch.setattr("flowmcp.rag.loader.Retriever", lambda *args, **kwargs: mock_retriever)

    rag = RAG("https://docs.example.com/faq")

    mock_get.assert_called_once_with(
        "https://docs.example.com/faq", follow_redirects=True, timeout=15.0
    )
    mock_retriever.add.assert_called_once()
    chunks = mock_retriever.add.call_args[0][0]
    combined = " ".join(chunks)
    assert "FAQ" in combined
    assert "30-day full refund." in combined
    assert "bad()" not in combined
    assert rag.retriever is mock_retriever
