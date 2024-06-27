import pytest
from transformers import AutoTokenizer

from chunking import Chunker, MIN_TOKENS


@pytest.fixture
def tokenizer():
    # Load a small tokenizer for testing purposes
    return AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")

@pytest.fixture
def chunker_semantic(tokenizer):
    # Create a Chunker instance with semantic strategy
    chunker = Chunker(
        chunking_strategy="semantic",
        tokenizer=tokenizer,
        buffer_size=1,
        breakpoint_percentile_threshold=0.98,
        embedding_model_name="jinaai/jina-embeddings-v2-small-en",
    )
    return chunker

@pytest.fixture
def chunker_fixed(tokenizer):
    # Create a Chunker instance with fixed strategy
    return Chunker(
        chunking_strategy="fixed",
        tokenizer=tokenizer,
        chunk_size=10,
    )

def test_chunk_semantically_small_text(chunker_semantic, tokenizer):
    text = "Short text"
    chunks = chunker_semantic.chunk_semantically(text)
    assert len(chunks) == 1
    assert chunks[0] == (0, len(tokenizer.encode(text, add_special_tokens=False)) - 1)

def test_chunk_semantically_long_text(chunker_semantic):
    text = """
    This is a longer text to test semantic chunking and should be split into chunks.
    
    Garden flowers encompass a wide variety of species cultivated for their aesthetic appeal and fragrance. 
    Common types include roses, known for their velvety petals and rich colors, and daisies, which have simple, 
    white petals with yellow centers. Sunflowers are popular for their large, 
    bright yellow blooms that follow the sun’s path across the sky.
    """
    chunks = chunker_semantic.chunk_semantically(text)
    assert len(chunks) > 1
    start, end = chunks[0]
    assert end - start + 1 >= MIN_TOKENS

def test_chunk_semantically_last_node_small(chunker_semantic):
    text = "This text is a small chunk."
    chunks = chunker_semantic.chunk_semantically(text)
    assert len(chunks) == 1
    start, end = chunks[0]
    assert end - start + 1 < MIN_TOKENS

def test_chunk_by_tokens_small_text(chunker_fixed):
    text = "Short text"
    chunks = chunker_fixed.chunk_by_tokens(text)
    assert len(chunks) == 1
    assert chunks[0] == (0, len(chunker_fixed.tokenizer.encode(text, add_special_tokens=False)) - 1)

def test_chunk_by_tokens_long_text(chunker_fixed):
    text = """
    This is a longer text to test semantic chunking and should be split into chunks.

    Garden flowers encompass a wide variety of species cultivated for their aesthetic appeal and fragrance. 
    Common types include roses, known for their velvety petals and rich colors, and daisies, which have simple, 
    white petals with yellow centers. Sunflowers are popular for their large, 
    bright yellow blooms that follow the sun’s path across the sky.
    """
    chunks = chunker_fixed.chunk_by_tokens(text)
    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start + 1 <= chunker_fixed.chunk_size

def test_chunk_by_tokens_exact_size_text(chunker_fixed):
    text = "This is a text with exactly one chunk."
    chunks = chunker_fixed.chunk_by_tokens(text)
    print(chunks)
    assert len(chunks) == 1
    assert chunks[0] == (0, 8)

def test_invalid_chunking_strategy(tokenizer):
    with pytest.raises(ValueError):
        Chunker(
            chunking_strategy="something",
            tokenizer=tokenizer,
        )