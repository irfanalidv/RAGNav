from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .models import Block, Document, RetrievalResult
from .retrieval import RAGNavIndex, RAGNavRetriever

try:
    __version__ = _pkg_version("ragnav")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.5.0"

__all__ = [
    "Block",
    "Document",
    "RetrievalResult",
    "RAGNavIndex",
    "RAGNavRetriever",
    "__version__",
]
