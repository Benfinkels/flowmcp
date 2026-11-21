
from unittest.mock import MagicMock, patch
from flowmcp.rag.loader import RAG


def test_rag_loads_and_chunks_url():
    html = (
        "<html><head><style>body{}</style></head>"
        "<body><h1>FAQ</h1><p>30-day full refund.</p><script>bad()</script></body></html>"
    )
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status.return_value = None

    mock_retriever = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get, patch(
        "flowmcp.rag.loader.Retriever", return_value=mock_retriever
    ):
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
