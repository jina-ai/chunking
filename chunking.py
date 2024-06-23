import bisect
import logging
from typing import List, Optional, Tuple, Union

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class Chunker:
    def __init__(
        self,
        chunking_strategy: str,
        tokenizer: Optional[Union[str, 'AutoTokenizer']] = None,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: float = 0.98,
        embedding_model_name: str = "jinaai/jina-embeddings-v2-small-en",
        chunk_size: int = 256,
    ):
        self.chunking_strategy = chunking_strategy
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                force_download=True,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = tokenizer
        if self.chunking_strategy == "semantic":
            self.buffer_size = buffer_size
            self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                max_length=512,
                trust_remote_code=True,
            )
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=self.buffer_size,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                embed_model=self.embed_model,
                show_progress=False,
            )
        elif self.chunking_strategy == "fixed":
            self.chunk_size = chunk_size
        else:
            raise ValueError("Unsupported chunking strategy")

    def chunk_semantically(
        self, text: str, span: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int, int]]:
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]
        # Tokenize the entire text
        tokens = self.tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=512,
            padding=True,
            truncation=True,
        )
        token_offsets = tokens.offset_mapping

        # Find the chunk indices for the start and end of the span
        start_token_index = (
            next(
                (i for i, offset in enumerate(token_offsets) if offset[0] >= span[0]),
                len(token_offsets),
            )
            if span
            else None
        )
        end_token_index = (
            next(
                (i for i, offset in enumerate(token_offsets) if offset[1] > span[1]),
                len(token_offsets),
            )
            if span
            else None
        )
        # check if relevant span is outside of the tokenized text (ie. truncated)
        # if it is, we don't want this sample
        if span:
            if start_token_index >= len(token_offsets) or end_token_index >= len(
                token_offsets
            ):
                return None

        chunks_and_labels = []
        for char_start, char_end in nodes:
            # convert char_start and char_end to token indices
            start_chunk_index = bisect.bisect_left(
                [offset[0] for offset in token_offsets], char_start
            )
            end_chunk_index = (
                bisect.bisect_right([offset[1] for offset in token_offsets], char_end)
                - 1
            )
            # if the chunk is outside of the tokenized text,
            # we don't want this sample
            # and we also don't want any chunks after that,
            # so break out of loop in this case
            if start_chunk_index >= len(token_offsets) or end_chunk_index >= len(
                token_offsets
            ):
                break
            # determine if this chunk contains the span
            if span:
                contains_span = (
                    start_chunk_index <= start_token_index <= end_chunk_index
                    or start_chunk_index <= end_token_index <= end_chunk_index
                )
                label = 1 if contains_span else 0
            else:
                label = None
            chunks_and_labels.append((start_chunk_index, end_chunk_index, label))

        return chunks_and_labels

    def chunk_by_tokens(
        self, text: str, span: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int, int]]:
        tokens = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping
        start_token_index = (
            bisect.bisect_left([offset[0] for offset in token_offsets], span[0])
            if span
            else None
        )
        end_token_index = (
            (bisect.bisect_right([offset[1] for offset in token_offsets], span[1]) - 1)
            if span
            else None
        )

        chunks_and_labels = []
        for i in range(0, len(token_offsets), self.chunk_size):
            chunk_end = min(i + self.chunk_size - 1, len(token_offsets) - 1)

            if span:
                contains_span = (
                    i <= start_token_index <= chunk_end
                    or i <= end_token_index <= chunk_end
                )
                label = 1 if contains_span else 0
            else:
                label = None
            chunks_and_labels.append((i, chunk_end, label))

        return chunks_and_labels

    def chunk(
        self,
        text: str,
        span: Optional[List[int]] = None,
        tokenizer: 'AutoTokenizer' = None,
    ):
        if tokenizer and not self.tokenizer:
            self.tokenizer = tokenizer
        if self.chunking_strategy == "semantic":
            if not self.tokenizer:
                self.tokenizer = tokenizer
            return self.chunk_semantically(text, span=span)
        elif self.chunking_strategy == "fixed":
            return self.chunk_by_tokens(text, span=span)
        else:
            raise ValueError("Unsupported chunking strategy")
