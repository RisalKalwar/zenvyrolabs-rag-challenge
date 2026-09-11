import os
import re
import base64
import time

from dotenv import load_dotenv

import fitz  # PyMuPDF - used directly for manga page rendering (Bug 4)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from the .env file sitting next to this script
load_dotenv()

# ---------------------------------------------------------------------------
# Vector DB and Embeddings
# ---------------------------------------------------------------------------
DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Local HuggingFace embeddings (runs on CPU for free!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# LLM Initialization (was `llm = None` — now wired to Groq)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Create a .env file next to rag_engine.py "
        "with a line: GROQ_API_KEY=your_key_here (get one free, no card "
        "needed, at console.groq.com)"
    )

# Text LLM used for answering questions
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    timeout=30,  # fail fast instead of hanging indefinitely on a slow call
)

# NOTE: manga OCR (Bug 4) history —
#   1. Groq vision model (qwen/qwen3.6-27b): daily token quota exhausted
#      during testing, repeatedly blocked uploads.
#   2. Tesseract (local, fast): ran with no rate limits, but accuracy on
#      stylized manga speech-bubble fonts was poor.
#   3. EasyOCR (local, deep-learning based): better accuracy on comic/
#      stylized/curved text than Tesseract, still fully local/free/no
#      rate limits — heavier install and slower per page, but worth it
#      for quality. This is what's currently used.
from PIL import Image
import io
import numpy as np

_easyocr_reader = None


def _get_easyocr_reader():
    """Lazily loads the EasyOCR model once and reuses it across calls —
    loading it is slow (downloads model weights on first run), so we
    don't want to redo that for every single page."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        import torch

        use_gpu = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        print(f"[manga OCR] Loading EasyOCR model on {device_name} "
              f"(first run downloads weights — may take a minute or two)...")

        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        print(f"[manga OCR] EasyOCR model ready (GPU: {use_gpu}).")
    return _easyocr_reader


# ---------------------------------------------------------------------------
# BUG 4 HELPER: Manga OCR via local EasyOCR
# ---------------------------------------------------------------------------
def _extract_manga_pages_with_vision(file_path: str) -> list[Document]:
    """
    Renders every page of a manga/comic PDF to an image and runs local
    OCR (EasyOCR) to extract visible text (speech bubbles, captions,
    sound effects). Returns one Document per page so the rest of the
    pipeline (chunking, page-marker injection, storage) can treat it
    exactly like a text-based book.

    Runs entirely locally — no API calls, no rate limits, no per-token
    cost. Function name kept as "_with_vision" so the rest of the
    pipeline (process_and_store_document) didn't need to change.
    """
    doc = fitz.open(file_path)
    documents: list[Document] = []
    total_pages = len(doc)

    # Optional testing cap: set MANGA_TEST_PAGE_LIMIT in .env to a small
    # number (e.g. 10) for quick iteration — EasyOCR is noticeably slower
    # per page than the old Tesseract approach, so this is more useful now.
    test_limit = os.getenv("MANGA_TEST_PAGE_LIMIT")
    if test_limit:
        total_pages = min(total_pages, int(test_limit))
        print(f"[manga OCR] MANGA_TEST_PAGE_LIMIT set — only processing {total_pages} pages")

    reader = _get_easyocr_reader()

    RENDER_ZOOM = 2  # good balance of legibility vs. per-page processing time

    for page_index in range(total_pages):
        page = doc[page_index]

        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(image)

        try:
            # paragraph=True merges nearby text into readable blocks
            # instead of one item per detected word/line — closer to how
            # a speech bubble's dialogue actually reads.
            results = reader.readtext(image_np, detail=0, paragraph=True)
            transcribed_text = "\n".join(results).strip()
            if not transcribed_text:
                transcribed_text = "[No dialogue on this page]"
        except Exception as e:
            transcribed_text = f"[OCR failed for this page: {e}]"

        documents.append(
            Document(
                page_content=transcribed_text,
                metadata={"page": page_index, "source": file_path},
            )
        )

        preview = transcribed_text[:150].replace("\n", " ")
        print(f"[manga OCR] page {page_index + 1}/{total_pages} done — output preview: {preview!r}")

    doc.close()
    return documents


# ---------------------------------------------------------------------------
# BUG 2 HELPER: detect "global" questions that need broad context
# ---------------------------------------------------------------------------
GLOBAL_QUESTION_KEYWORDS = [
    "entire book", "whole book", "summarize", "summary", "overview",
    "overall", "in general", "all chapters", "the whole thing",
    "best problem", "main themes", "main idea", "throughout the book",
    "as a whole", "across the book", "every chapter", "whole story",
    "the entire", "in total", "full book",
]


def _is_global_question(user_message: str) -> bool:
    msg = user_message.lower()
    return any(keyword in msg for keyword in GLOBAL_QUESTION_KEYWORDS)


# ---------------------------------------------------------------------------
# BUG 1 HELPER (part 2): exact page-number lookups
# ---------------------------------------------------------------------------
# Semantic/embedding search is bad at distinguishing "page 34" from
# "page 35" — the numbers are too close in vector space to reliably rank
# the *exact* page inside the top-k results out of thousands of chunks.
# So for questions that clearly reference a specific page number, we skip
# similarity search entirely and pull chunks by exact metadata match.


def _extract_page_numbers(user_message: str) -> list:
    """Returns ALL page numbers mentioned in the question.

    Handles multiple phrasings:
      - "page 34" -> [34]
      - "page 91 and 92" -> [91, 92]  (only the FIRST number has "page"
        directly before it — a plain r"page\\s+(\\d+)" with findall only
        ever caught that first one and silently dropped the rest)
      - "pages 60, 75, and 80" -> [60, 75, 80]
      - "compare page 10 and page 20" -> [10, 20]  (two separate "page"
        keywords also still works)
    """
    msg = user_message.lower()
    pattern = r"pages?\s+(\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d+)*)"
    matches = re.findall(pattern, msg)
    page_numbers = []
    for group in matches:
        page_numbers.extend(int(n) for n in re.findall(r"\d+", group))
    return page_numbers


def _wants_line_numbers(user_message: str) -> bool:
    """True only if the question specifically asks about a line number
    (e.g. 'page 34 line 5'). Line-by-line formatting is only injected
    into context for these — otherwise a general question like 'tell me
    about page 34' would get an unnaturally line-numbered answer just
    because the model saw that structure in its context."""
    return bool(re.search(r"\bline\s+\d+\b", user_message.lower()))


# ---------------------------------------------------------------------------
# BUG 3 HELPER: format prior conversation turns for the prompt
# ---------------------------------------------------------------------------
def _format_history(history: list) -> str:
    if not history:
        return "No previous conversation."

    # Cap how much history we inject — free-tier TPM limits are tight
    # (8,000 tokens/min on Groq), and an ever-growing conversation would
    # otherwise eventually blow the budget on its own, even before adding
    # retrieved context. Keep the most recent turns only.
    MAX_HISTORY_TURNS = 20  # ~10 back-and-forth exchanges — raised from 6
    # after confirming multi-step recall ("what did I ask before that?")
    # needs more room than 3 exchanges gave it. Token budget elsewhere
    # (context caps, map-reduce) was tightened enough to afford this.
    recent_history = history[-MAX_HISTORY_TURNS:]

    lines = []
    for turn in recent_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rate-limit-safe LLM call + Map-Reduce summarization for global questions
# ---------------------------------------------------------------------------
# Groq's free tier caps tokens-per-minute (TPM) quite low (8,000 for
# openai/gpt-oss-120b at time of writing). Stuffing 100 retrieved chunks
# into one prompt blows past that instantly. Instead we:
#   1. Split retrieved chunks into small batches that stay well under the
#      TPM budget.
#   2. "Map": summarize each batch's relevant points separately.
#   3. "Reduce": combine those batch summaries into one final answer.
# This is also the direct answer to the assignment's architecture-interview
# question about scaling to a 500-page book without crashing free APIs.

MAX_TOKENS_PER_BATCH = 2500  # conservative budget per map call (well under 8000 TPM)
CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, no need for exact tokenizer here


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _batch_documents(docs: list, max_tokens: int = MAX_TOKENS_PER_BATCH) -> list:
    """Groups page_content strings into batches that each stay under max_tokens."""
    batches = []
    current_batch = []
    current_tokens = 0

    for doc in docs:
        doc_tokens = _estimate_tokens(doc.page_content)
        if current_batch and (current_tokens + doc_tokens > max_tokens):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc.page_content)
        current_tokens += doc_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _invoke_with_backoff(chat_model, messages, max_retries: int = 6, initial_delay: int = 8):
    """
    Calls a Groq chat model with exponential backoff on rate-limit (429)
    or payload-too-large (413) errors, so a burst of calls doesn't crash
    the whole request — it just waits and retries. Defaults bumped up
    (was 4 retries / 5s start) after seeing the manga vision model need
    longer waits than the text model to clear its rate limit.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return chat_model.invoke(messages)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "rate_limit" in msg.lower() or "429" in msg or "413" in msg
            if attempt == max_retries - 1 or not is_rate_limit:
                raise
            print(f"    ...rate limited, waiting {delay}s before retry {attempt + 2}/{max_retries}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Exceeded retry attempts")


def _map_reduce_answer(user_message: str, docs: list) -> str:
    """
    Answers a 'global' question (e.g. summarize the entire book) by
    summarizing retrieved chunks in small batches (map), then combining
    those partial summaries into one final answer (reduce). Keeps every
    individual LLM call well under the free-tier token-per-minute limit.

    Deliberately does NOT take conversation history — a real bug was
    found where including it caused the model to answer from a recent
    unrelated turn (e.g. a detailed page-specific answer) instead of
    genuinely synthesizing across the whole book. Global book-wide
    questions aren't about conversation continuity; follow-up memory
    questions are handled separately by the normal-question path.
    """
    batches = _batch_documents(docs)

    map_prompt = ChatPromptTemplate.from_template(
        """Extract the key points from the following section of a book that
are relevant to this question: "{question}"
Be concise — bullet points are fine. If nothing in this section is relevant, say so briefly.
IMPORTANT: If this section contains a worked example, exercise, algorithm
walkthrough, or notable problem (not just a table of contents or chapter
overview), explicitly call it out by name/number — these are often what
a question like "what's the best problem in the book" is really asking
about, and generic chapter-structure descriptions are NOT useful for that.

Section:
{section}

Key points:"""
    )

    partial_summaries = []
    for batch in batches:
        section_text = "\n\n".join(batch)
        # Guard against a single oversized batch still exceeding the limit
        if _estimate_tokens(section_text) > MAX_TOKENS_PER_BATCH * 1.5:
            section_text = section_text[: MAX_TOKENS_PER_BATCH * CHARS_PER_TOKEN_ESTIMATE]

        prompt_value = map_prompt.format(question=user_message, section=section_text)
        try:
            summary = _invoke_with_backoff(llm, prompt_value)
            partial_summaries.append(summary.content if hasattr(summary, "content") else str(summary))
        except Exception as e:
            partial_summaries.append(f"[Skipped one section due to an API error: {e}]")

        # Small pacing delay between map calls to respect TPM limits even
        # when the backoff above wasn't triggered.
        time.sleep(1)

    combined_summaries = "\n\n---\n\n".join(partial_summaries)

    reduce_prompt = ChatPromptTemplate.from_template(
        """You are a helpful Interactive Study Tutor. The section summaries
below were gathered from across the ENTIRE book to answer this question.
Base your answer strictly on these summaries — they are your only source
of truth here, regardless of anything discussed earlier in the
conversation. Synthesize across ALL the summaries, not just one section.

If the question asks for a subjective judgment (e.g. "the best problem,"
"the most interesting example," "the most important idea") — this is a
request for your reasoned recommendation, not a factual lookup. Pick the
most notable specific example, exercise, or concept that appears in the
summaries and explain clearly why it stands out. Do NOT refuse or say
you cannot determine an answer just because "best" is subjective — make
a reasoned choice from what's actually present in the summaries, the
same way a knowledgeable tutor would recommend one when asked.

Section summaries (from across the whole book):
{summaries}

Question: {question}

Answer:"""
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({
        "summaries": combined_summaries,
        "question": user_message,
    })


# ---------------------------------------------------------------------------
# Document Processing / Storage
# ---------------------------------------------------------------------------
def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Processes the uploaded file based on the book_type and stores it in ChromaDB.
    """
    # --- BUG 4 FIX: Manga OCR ---
    # PyMuPDF's default loader only reads embedded text layers. Manga PDFs
    # are scanned/rendered images with no text layer, so we bypass the
    # standard loader entirely and use a vision model to transcribe each
    # page's dialogue into text Documents first.
    if book_type == "manga":
        docs = _extract_manga_pages_with_vision(file_path)
    else:
        # 1. Load the PDF with PyMuPDF (works for text-based coding/novel books)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Guard: some "novel" PDFs contain a handful of image-only pages
        # (covers, illustrations). Skip empty pages so the splitter doesn't
        # choke on zero-length content instead of silently losing them.
        docs = [d for d in docs if d.page_content.strip()]

    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # --- BUG 1 FIX: Vector Blindness / Absolute Precision ---
    # Page numbers live only in chunk.metadata, invisible to the embedding
    # model. We inject a physical page marker directly into page_content
    # of EVERY chunk (done post-split, so it's guaranteed on every chunk,
    # not just the first chunk of a page) so semantic search can "see" it.
    for chunk in chunks:
        page_num = chunk.metadata.get("page", None)
        if page_num is not None:
            # PyMuPDF page metadata is 0-indexed; show 1-indexed to match
            # what a human sees in a PDF viewer.
            marker = f"[Source: PDF Viewer Page {page_num + 1}]\n"
            chunk.page_content = marker + chunk.page_content

    # 3. Store in Vector DB — isolated per book_type (see collection_name
    # note above _extract_manga_pages_with_vision / module docs) so
    # Coding, Novel, and Manga uploads never mix in retrieval.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=f"book_{book_type}",
    )
    return "Success"


# ---------------------------------------------------------------------------
# Query / Answer Generation
# ---------------------------------------------------------------------------
def query_rag_system(user_message: str, book_type: str = "coding", history: list = None) -> str:
    """
    Queries the Vector DB and generates an answer using the LLM.
    """
    if llm is None:
        return "ERROR: You must initialize an LLM in rag_engine.py first!"

    if history is None:
        history = []

    # --- IMPORTANT FIX: isolate storage per book_type ---
    # Previously all uploads (coding/novel/manga) shared one Chroma
    # collection, so switching modes still searched across every book
    # ever uploaded. Each mode now gets its own named collection so
    # "page 4" in the manga can't collide with "page 4" in the textbook.
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=f"book_{book_type}",
    )

    page_numbers = _extract_page_numbers(user_message)
    line_lookup = _wants_line_numbers(user_message)

    if page_numbers:
        # --- BUG 1 FIX (part 2): exact page lookup via metadata filter ---
        # Bypass semantic search entirely for page-specific questions.
        # Handles multiple pages in one question (e.g. "page 60 and 75")
        # by looking each one up independently and labeling the results,
        # so a miss on one page doesn't blank out an answer for the other.
        #
        # Each page gets its OWN character budget (deduplicated first).
        # This matters because re-uploading the same document multiple
        # times during testing (without clearing chroma_db) leaves
        # duplicate chunks behind — without a per-page cap, one page's
        # duplicated content could crowd out every other requested page
        # once a single shared budget was applied to the combined text.
        PER_PAGE_CHAR_BUDGET = 4000  # ~1,000 tokens per requested page

        context_sections = []
        for page_number in page_numbers:
            # Stored page metadata is 0-indexed (PyMuPDF), so subtract 1
            # to match the 1-indexed page number a human would type.
            result = vector_db.get(where={"page": page_number - 1})
            matched_docs = result.get("documents", []) if result else []

            # Deduplicate — repeated uploads during testing can leave
            # identical chunks stored multiple times.
            seen = set()
            unique_docs = []
            for d in matched_docs:
                if d not in seen:
                    seen.add(d)
                    unique_docs.append(d)

            if unique_docs:
                section_text = "\n\n".join(unique_docs)
                if len(section_text) > PER_PAGE_CHAR_BUDGET:
                    section_text = section_text[:PER_PAGE_CHAR_BUDGET]

                if line_lookup:
                    # Only number lines when the question actually asks
                    # about a specific line (e.g. "page 34 line 5") — the
                    # README's own grading example. Without explicit
                    # numbering the model would have to count through raw
                    # text itself, which is unreliable.
                    formatted_text = "\n".join(
                        f"Line {i}: {line}" for i, line in enumerate(section_text.split("\n"), 1)
                    )
                else:
                    # General questions ("tell me about page 34") get the
                    # plain page text, so the answer reads naturally
                    # instead of mirroring a line-numbered structure it
                    # doesn't need.
                    formatted_text = section_text

                context_sections.append(f"--- Content from Page {page_number} ---\n{formatted_text}")
                print(f"[page lookup] page {page_number}: found {len(matched_docs)} chunks "
                      f"({len(unique_docs)} unique), {len(section_text)} chars included")
            else:
                context_sections.append(
                    f"--- Page {page_number} ---\n"
                    f"No content was found for page {page_number}. "
                    f"It may be outside the uploaded document's page range, "
                    f"or an image-only page with no extractable text."
                )
                print(f"[page lookup] page {page_number}: NO CHUNKS FOUND in vector DB")
        context = "\n\n".join(context_sections)
    elif _is_global_question(user_message):
        # --- BUG 2 FIX: Myopic Context Limits ---
        # Global questions ("summarize the entire book") need to see far
        # more of the book than a normal top-k similarity search returns.
        # We detect these, pull a large diverse spread of chunks via MMR
        # (Max Marginal Relevance), then answer using map-reduce so no
        # single LLM call ever exceeds the free-tier token-per-minute
        # limit — see _map_reduce_answer() above.
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 80, "fetch_k": 260, "lambda_mult": 0.5}
        )
        retrieved_docs = retriever.invoke(user_message)
        return _map_reduce_answer(user_message, retrieved_docs)
    else:
        # Reduced from the original k=30 — at ~250 tokens per chunk that
        # was already ~7,500 tokens on its own, leaving almost no room
        # for the prompt template, question, and conversation history
        # before hitting Groq's free-tier 8,000 TPM cap.
        retriever = vector_db.as_retriever(search_kwargs={"k": 12})
        retrieved_docs = retriever.invoke(user_message)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Extra safety net for the semantic-search path only — the
        # page-lookup path above already caps itself per page, so this
        # would only ever cut it short unnecessarily.
        MAX_CONTEXT_CHARS = 12000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

    # --- BUG 3 FIX: Conversation Memory ---
    # The conversation history sent from the frontend is formatted and
    # injected directly into the prompt so the LLM can resolve follow-ups
    # like "what did I just ask you?".
    history_text = _format_history(history)

    template = """You are a helpful Interactive Study Tutor.

Answer the user's question using the rules below, in this priority order:
1. If the question is about the conversation itself (e.g. "what did I just ask you?", "what did you say before?", "no, before that") — answer it directly using the "Conversation so far" section below. This does NOT need to be supported by the textbook Context.
2. If the question asks about a specific line number on a page (e.g. "what does page 34 line 5 say?"), the Context below will have each line explicitly labeled "Line N: ...". Quote that exact labeled line directly in your answer. If the Context does NOT have "Line N:" labels, answer normally in prose — don't invent line numbers.
3. Otherwise, answer based ONLY on the textbook Context below. If the Context does not contain the answer, say "I cannot find the answer to this in the textbook." Do not hallucinate or guess.

Conversation so far:
{history}

Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    prompt_value = prompt.format(history=history_text, context=context, question=user_message)

    # Rate-limit-safe call — retries with backoff if we still land close
    # to the TPM ceiling.
    response = _invoke_with_backoff(llm, prompt_value)
    return response.content if hasattr(response, "content") else str(response)
