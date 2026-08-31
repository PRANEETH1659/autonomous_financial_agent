"""Chunking + the BM25 keyword-search index that replaced the old
ChromaDB/embeddings pipeline. store_in_vector_db's punctuation-only-chunk
filtering is the load-bearing edge case here - BM25Okapi divides by the
corpus's average document length, so a corpus that tokenizes to nothing
raises ZeroDivisionError if that filter regresses."""
import pytest

import tools


class TestChunkText:
    def test_short_text_produces_a_single_chunk(self):
        chunks = tools.chunk_text("A short paragraph about Acme Corp.")
        assert len(chunks) == 1
        assert "Acme Corp" in chunks[0]

    def test_long_text_is_split_into_multiple_overlapping_chunks(self):
        long_text = " ".join(f"sentence{i} has some words in it." for i in range(200))
        chunks = tools.chunk_text(long_text)
        assert len(chunks) > 1
        # chunk_size=500 is a soft target for the splitter, not a hard cap on
        # every chunk - assert nothing has wildly exceeded it.
        assert all(len(c) <= 600 for c in chunks)

    def test_empty_text_produces_no_unusable_chunks(self):
        chunks = tools.chunk_text("")
        assert chunks == [] or all(c.strip() == "" for c in chunks)


class TestTokenize:
    def test_lowercases_and_splits_on_word_boundaries(self):
        assert tools._tokenize("Apple's Q3 Revenue: $4.2B!") == [
            "apple", "s", "q3", "revenue", "4", "2b",
        ]

    def test_punctuation_only_text_tokenizes_to_nothing(self):
        assert tools._tokenize("... --- !!!") == []


class TestStoreInVectorDb:
    def test_empty_chunk_list_returns_none(self):
        assert tools.store_in_vector_db([], collection_name="c") is None
        assert "c" not in tools._chunk_store

    def test_punctuation_only_chunks_are_filtered_and_return_none(self):
        # This is the exact regression this filter guards against: a PDF
        # page of only punctuation/whitespace must not reach BM25Okapi,
        # which would divide by a zero average document length.
        result = tools.store_in_vector_db(["...", "---", "   "], collection_name="c")
        assert result is None
        assert "c" not in tools._chunk_store

    def test_mixed_usable_and_unusable_chunks_keeps_only_usable_ones(self):
        result = tools.store_in_vector_db(
            ["Revenue grew 20% year over year.", "...", "Net income was $4.2B."],
            collection_name="c",
        )
        assert result is not None
        assert result["chunks"] == [
            "Revenue grew 20% year over year.",
            "Net income was $4.2B.",
        ]

    def test_normal_chunks_are_stored_and_retrievable_by_collection_name(self):
        chunks = ["Apple makes iPhones.", "Tesla makes electric cars."]
        stored = tools.store_in_vector_db(chunks, collection_name="my_collection")
        assert stored is tools._chunk_store["my_collection"]
        assert stored["chunks"] == chunks


class TestRetrieveContext:
    def test_no_collection_stored_returns_friendly_message(self):
        assert tools.retrieve_context("anything", collection_name="never_stored") == "No data stored yet."

    def test_k_larger_than_available_chunks_does_not_crash(self):
        tools.store_in_vector_db(["Only one chunk here."], collection_name="c")
        result = tools.retrieve_context("chunk", k=10, collection_name="c")
        assert "Only one chunk here." in result

    def test_most_relevant_chunk_ranks_first(self):
        tools.store_in_vector_db(
            [
                "Bananas are yellow fruit.",
                "Apple stock price rose sharply on strong iPhone sales.",
                "The weather today is sunny.",
            ],
            collection_name="c",
        )
        result = tools.retrieve_context("Apple iPhone sales", k=1, collection_name="c")
        assert "Apple stock price rose sharply" in result

    def test_multiple_results_joined_with_separator(self):
        tools.store_in_vector_db(
            ["Apple news one.", "Apple news two.", "Unrelated banana news."],
            collection_name="c",
        )
        result = tools.retrieve_context("Apple news", k=2, collection_name="c")
        assert result.count("\n---\n") == 1

    def test_internal_error_is_reported_not_raised(self):
        tools.store_in_vector_db(["Some content here."], collection_name="c")

        class ExplodingBM25:
            def get_scores(self, *_):
                raise RuntimeError("index corrupted")

        tools._chunk_store["c"]["bm25"] = ExplodingBM25()
        result = tools.retrieve_context("anything", collection_name="c")
        assert "Error retrieving context" in result
        assert "index corrupted" in result


class TestClearVectorDb:
    def test_removes_a_stored_collection(self):
        tools.store_in_vector_db(["Some content."], collection_name="c")
        tools.clear_vector_db("c")
        assert tools.retrieve_context("content", collection_name="c") == "No data stored yet."

    def test_clearing_a_collection_that_was_never_stored_does_not_raise(self):
        tools.clear_vector_db("never_existed")  # should be a silent no-op
