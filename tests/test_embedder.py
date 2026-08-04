import numpy as np

from rag_eval.retrieval import embedder


class _FakeEmbeddingResponse:
    def __init__(self, values):
        self.embeddings = [type("Embedding", (), {"values": values})()]


class _FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        return _FakeEmbeddingResponse([0.6, 0.8])


class _FakeClient:
    def __init__(self, api_key=None):
        self.models = _FakeModels()
        self.api_key = api_key


class _FailingModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        raise RuntimeError("boom")


class _FailingClient:
    def __init__(self, api_key=None):
        self.models = _FailingModels()
        self.api_key = api_key


class _FakeSentenceTransformerEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def dimension(self):
        return 3

    def embed_query(self, query):
        return np.array([0.0, 1.0], dtype=np.float32)


def test_gemini_embedder_uses_models_api(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(embedder, "Client", _FakeClient)

    emb = embedder.GeminiEmbedder()
    vecs = emb.embed_documents(["hello"])

    assert vecs.shape == (1, 2)
    assert np.allclose(vecs[0], np.array([0.6, 0.8], dtype=np.float32) / np.linalg.norm([0.6, 0.8]))


def test_build_embedder_falls_back_when_gemini_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", _FakeSentenceTransformerEmbedder)

    emb = embedder.build_embedder({"provider": "gemini"})

    assert isinstance(emb, _FakeSentenceTransformerEmbedder)


def test_gemini_embedder_falls_back_on_client_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(embedder, "Client", _FailingClient)
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", _FakeSentenceTransformerEmbedder)

    emb = embedder.GeminiEmbedder()
    vec = emb.embed_query("hello")

    assert vec.shape == (2,)


def test_build_embedder_accepts_local_provider_alias(monkeypatch):
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", _FakeSentenceTransformerEmbedder)

    emb = embedder.build_embedder({"provider": "local", "model": "all-MiniLM-L6-v2"})

    assert isinstance(emb, _FakeSentenceTransformerEmbedder)
    assert emb.kwargs["model_name"] == "all-MiniLM-L6-v2"


def test_gemini_embedder_supports_embed_compatibility_method(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(embedder, "Client", _FakeClient)

    emb = embedder.GeminiEmbedder()
    vecs = emb.embed(["hello"])

    assert vecs.shape == (1, 2)
