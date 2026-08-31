"""Deterministic paragraph-first chunking for the local corpus."""

import re
from dataclasses import dataclass
from typing import Any

from src.corpus import find_document, load_documents, read_document


DEFAULT_MAX_WORDS = 300
DEFAULT_OVERLAP_WORDS = 50

_PARAGRAPH_BREAK = re.compile(r"\n(?:[ \t]*\n)+")
_SENTENCE_BREAK = re.compile(r"[.!?](?:[»”\"])?([ \t\n]+)(?=\S)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_document_id: str
    work_title: str
    chapter_number: int
    chapter_title: str
    chunk_position: int
    text: str
    word_count: int
    source_url: str


@dataclass(frozen=True)
class _Piece:
    """A sentence-sized source span and its exact preceding separator."""

    text: str
    paragraph: int
    separator_before: str
    word_count: int


def count_words(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[tuple[str, str]]:
    """Return (paragraph, preceding separator) pairs without rewriting text."""
    content = text.strip("\n")
    if not content:
        return []

    paragraphs: list[tuple[str, str]] = []
    start = 0
    separator = ""
    for match in _PARAGRAPH_BREAK.finditer(content):
        paragraph = content[start : match.start()]
        if paragraph:
            paragraphs.append((paragraph, separator))
        separator = match.group(0)
        start = match.end()

    final_paragraph = content[start:]
    if final_paragraph:
        paragraphs.append((final_paragraph, separator))
    return paragraphs


def _split_long_piece(piece: _Piece, max_words: int) -> list[_Piece]:
    """Use exact whitespace boundaries if a single sentence exceeds the limit."""
    if piece.word_count <= max_words:
        return [piece]

    words = list(re.finditer(r"\S+", piece.text))
    result: list[_Piece] = []
    previous_end = 0
    for offset in range(0, len(words), max_words):
        group = words[offset : offset + max_words]
        start = group[0].start()
        end = group[-1].end()
        separator = (
            piece.separator_before
            if not result
            else piece.text[previous_end:start]
        )
        span = piece.text[start:end]
        result.append(
            _Piece(
                text=span,
                paragraph=piece.paragraph,
                separator_before=separator,
                word_count=len(group),
            )
        )
        previous_end = end
    return result


def _sentence_pieces(
    paragraph: str,
    paragraph_number: int,
    paragraph_separator: str,
    max_words: int,
) -> list[_Piece]:
    """Split a paragraph at deterministic sentence boundaries."""
    pieces: list[_Piece] = []
    start = 0
    separator = paragraph_separator

    for match in _SENTENCE_BREAK.finditer(paragraph):
        whitespace_start, whitespace_end = match.span(1)
        sentence = paragraph[start:whitespace_start]
        if sentence:
            piece = _Piece(
                text=sentence,
                paragraph=paragraph_number,
                separator_before=separator,
                word_count=count_words(sentence),
            )
            pieces.extend(_split_long_piece(piece, max_words))
        separator = paragraph[whitespace_start:whitespace_end]
        start = whitespace_end

    final_sentence = paragraph[start:]
    if final_sentence:
        piece = _Piece(
            text=final_sentence,
            paragraph=paragraph_number,
            separator_before=separator,
            word_count=count_words(final_sentence),
        )
        pieces.extend(_split_long_piece(piece, max_words))
    return pieces


def _source_pieces(
    text: str, max_words: int
) -> tuple[list[_Piece], dict[int, tuple[int, int, int]]]:
    """Build sentence pieces plus paragraph (start, end, words) ranges."""
    pieces: list[_Piece] = []
    paragraph_ranges: dict[int, tuple[int, int, int]] = {}

    for number, (paragraph, separator) in enumerate(_split_paragraphs(text)):
        start = len(pieces)
        pieces.extend(_sentence_pieces(paragraph, number, separator, max_words))
        end = len(pieces)
        paragraph_ranges[number] = (start, end, count_words(paragraph))
    return pieces, paragraph_ranges


def _render_span(
    pieces: list[_Piece], start: int, start_offset: int, end: int
) -> str:
    text = pieces[start].text[start_offset:]
    for piece in pieces[start + 1 : end]:
        text += piece.separator_before + piece.text
    return text


def _required_new_words(
    pieces: list[_Piece],
    paragraph_ranges: dict[int, tuple[int, int, int]],
    cursor: int,
    max_words: int,
) -> int:
    piece = pieces[cursor]
    paragraph_start, _, paragraph_words = paragraph_ranges[piece.paragraph]
    if cursor == paragraph_start and paragraph_words <= max_words:
        return paragraph_words
    return piece.word_count


def _choose_overlap_start(
    pieces: list[_Piece],
    end: int,
    overlap_target: int,
    capacity: int,
    paragraph_ranges: dict[int, tuple[int, int, int]],
) -> tuple[int, int]:
    """Choose the source suffix closest to the target without blocking progress."""
    if overlap_target == 0 or capacity == 0:
        return end, 0

    best_start = end
    best_score: tuple[int, int, int] | None = None
    words = 0
    for start in range(end - 1, -1, -1):
        words += pieces[start].word_count
        if words > capacity:
            break
        paragraph_start = paragraph_ranges[pieces[start].paragraph][0]
        score = (
            abs(words - overlap_target),
            0 if start == paragraph_start else 1,
            -words,
        )
        if best_score is None or score < best_score:
            best_start = start
            best_score = score
    if best_start != end:
        return best_start, 0

    # If no complete sentence fits, keep a smaller exact word-boundary suffix.
    # This is the last resort that makes overlap possible beside a large paragraph.
    last_piece = pieces[end - 1]
    words = list(re.finditer(r"\S+", last_piece.text))
    take = min(overlap_target, capacity, len(words))
    if take == 0:
        return end, 0
    return end - 1, words[-take].start()


def chunk_text(
    document: dict[str, Any],
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Generate ordered, deterministic chunks for one source document."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between zero and max_words - 1")

    pieces, paragraph_ranges = _source_pieces(text, max_words)
    if not pieces:
        return []

    spans: list[tuple[int, int, int]] = []
    cursor = 0
    overlap_start = (0, 0)

    while cursor < len(pieces):
        start, start_offset = overlap_start
        words = (
            count_words(_render_span(pieces, start, start_offset, cursor))
            if start < cursor
            else 0
        )
        end = cursor

        while end < len(pieces):
            piece = pieces[end]
            paragraph_start, paragraph_end, paragraph_words = paragraph_ranges[
                piece.paragraph
            ]
            if end == paragraph_start and paragraph_words <= max_words:
                next_end = paragraph_end
                added_words = paragraph_words
            else:
                next_end = end + 1
                added_words = piece.word_count

            if words + added_words > max_words:
                break
            words += added_words
            end = next_end

        if end == cursor:
            # Overlap is optional; source progress and the hard maximum are not.
            start = cursor
            start_offset = 0
            overlap_start = (cursor, 0)
            words = 0
            continue

        spans.append((start, start_offset, end))
        if end == len(pieces):
            break

        required_words = _required_new_words(
            pieces, paragraph_ranges, end, max_words
        )
        overlap_start = _choose_overlap_start(
            pieces,
            end,
            overlap_words,
            max_words - required_words,
            paragraph_ranges,
        )
        cursor = end

    chunks: list[Chunk] = []
    for position, (start, start_offset, end) in enumerate(spans, start=1):
        chunk = _render_span(pieces, start, start_offset, end)
        chunks.append(
            Chunk(
                chunk_id=f'{document["id"]}-chunk-{position:03}',
                source_document_id=document["id"],
                work_title=document["work_title"],
                chapter_number=document["chapter_number"],
                chapter_title=document["chapter_title"],
                chunk_position=position,
                text=chunk,
                word_count=count_words(chunk),
                source_url=document["source_url"],
            )
        )
    return chunks


def chunk_document(
    document: dict[str, Any],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    return chunk_text(document, read_document(document), max_words, overlap_words)


def load_chunks(
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents():
        chunks.extend(chunk_document(document, max_words, overlap_words))
    return chunks


def find_chunk(chunk_id: str) -> Chunk:
    for chunk in load_chunks():
        if chunk.chunk_id == chunk_id:
            return chunk
    raise KeyError(chunk_id)


def chunks_for_document(document_id: str) -> list[Chunk]:
    return chunk_document(find_document(document_id))


def corpus_chunk_stats() -> dict[str, int | float]:
    chunks = load_chunks()
    sizes = [chunk.word_count for chunk in chunks]
    return {
        "source_documents": len(load_documents()),
        "total_chunks": len(chunks),
        "minimum_words": min(sizes),
        "maximum_words": max(sizes),
        "average_words": sum(sizes) / len(sizes),
        "configured_maximum": DEFAULT_MAX_WORDS,
        "overlap_target": DEFAULT_OVERLAP_WORDS,
    }
