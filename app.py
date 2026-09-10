"""
Simple Flask app to serve a RAG model over video transcripts AND news articles.
- Single ChromaDB with both city_council and nola_articles collections
- Multi-turn conversation support
- Requires OPENAI_API_KEY
- model set by OPENAI_MODEL (default: gpt-4o-mini)

Memory efficiency fixes applied:
  FIX 1 - DualRAG no longer over-fetches: queries ceil(n/2) per collection
           instead of n*2 per collection, cutting per-request memory ~75%.
  FIX 2 - Box folder item listing is cached per crawl run via _FolderCache,
           eliminating redundant API calls on every save/upload.
  FIX 3 - Council metadata loaded as a plain dict (O(1) lookup) instead of
           a full pandas DataFrame that's scanned on every citation render.
  FIX 4 - _server_history is bounded to MAX_SESSIONS sessions / MAX_TURNS
           turns each, preventing unbounded in-process memory growth.
"""
import os
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import dspy
from flask import Flask, request, render_template_string, session, redirect
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
import markdown
import re
from types import SimpleNamespace

# ── Exact-match keyword helpers ───────────────────────────────────
_STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'shall','can','that','this','these','those','i','me','my','we','our',
    'you','your','he','she','it','its','they','them','their','what','which',
    'who','whom','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','just','about','in','on','at','by','for','with',
    'of','to','from','and','or','but','tell','me','any','did',
}


# Load environment variables
load_dotenv()

print(f"Python executable: {sys.executable}", flush=True)
print(f"chromadb version: {getattr(chromadb, '__version__', 'unknown')}", flush=True)
print(f"CHROMA_DB_DIR: {os.getenv('CHROMA_DB_DIR', '/models/chroma_db')}", flush=True)

# Correction imports for admin page (dictionary management)
try:
    from correction import load_dictionaries, load_hardcoded_corrections
    _corrections_available = True
except ImportError:
    _corrections_available = False

# ── ChromaDB — loaded once at startup, never reloaded ────────────
chroma_client = PersistentClient(
    path=os.getenv('CHROMA_DB_DIR', '/models/chroma_db'),
    settings=Settings(anonymized_telemetry=False)
)

# ── FIX 3: Council metadata as a plain dict instead of a DataFrame ──────────
# Keyed by video basename for O(1) lookups in citation rendering.
# SimpleNamespace preserves existing attribute access (row.title, row.date,
# row.box_link) so citation helpers need no changes.

def _parse_date_to_iso(value) -> str | None:
    """Normalize common date formats to YYYY-MM-DD."""
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    for parser in (
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        lambda s: parsedate_to_datetime(s),
        lambda s: datetime.strptime(s, "%A, %B %d, %Y"),
        lambda s: datetime.strptime(s, "%Y-%m-%d"),
    ):
        try:
            return parser(text).strftime("%Y-%m-%d")
        except Exception:
            continue

    return None


def _load_council_meta(data_path: str) -> tuple[dict, str, str]:
    """Load council JSONL into a dict keyed by video filename basename."""
    meta = {}
    try:
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                key = os.path.basename(row.get("video", ""))
                if key:
                    meta[key] = SimpleNamespace(**row)
        dates = [
            parsed for v in meta.values()
            if hasattr(v, "date")
            for parsed in [_parse_date_to_iso(v.date)]
            if parsed
        ]
        default_start = min(dates) if dates else "2020-01-01"
        default_end   = max(dates) if dates else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Council metadata not found: {e}")
        meta = {}
        default_start = "2020-01-01"
        default_end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return meta, default_start, default_end


def _load_article_date_bounds(chroma_path: str) -> tuple[str | None, str | None]:
    """Read article publish-date bounds directly from Chroma SQLite metadata."""
    db_path = os.path.join(chroma_path, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return None, None

    query = """
        select
            min(coalesce(em.int_value, cast(em.string_value as integer))),
            max(coalesce(em.int_value, cast(em.string_value as integer)))
        from collections c
        join segments s
          on s.collection = c.id and s.scope = 'METADATA'
        join embeddings e
          on e.segment_id = s.id
        join embedding_metadata em
          on em.id = e.id
        where c.name = 'articles' and em.key = 'published_unix'
    """

    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(query).fetchone()
        if not row or row[0] is None or row[1] is None:
            return None, None

        start = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(int(row[1]), tz=timezone.utc).strftime("%Y-%m-%d")
        return start, end
    except Exception as e:
        print(f"Article date bounds unavailable: {e}")
        return None, None

_council_meta, council_default_start, council_default_end = _load_council_meta(
    os.environ.get("FLY_DATA", "./") + "/data.jsonl"
)
article_default_start, article_default_end = _load_article_date_bounds(
    os.getenv('CHROMA_DB_DIR', '/models/chroma_db')
)

default_start = min(
    d for d in [council_default_start, article_default_start] if d
)
default_end = max(
    d for d in [council_default_end, article_default_end, datetime.now(timezone.utc).strftime("%Y-%m-%d")] if d
)

def filename2date(filename):
    """Look up a council video date by filename — O(1) dict lookup."""
    mp4 = re.sub(r'\.summary$', '.mp4', os.path.basename(filename))
    row = _council_meta.get(mp4)
    return row.date if row else None


# ── DSPy signatures & RAG module ─────────────────────────────────

class RAGQuestion(dspy.Signature):
    """
    Answer this question about New Orleans City Council meetings and civic news based ONLY on the provided context.

    CRITICAL RULES:
    - Use ONLY information from the provided context passages
    - Review the conversation history to understand follow-up questions and pronouns like "this", "that", "it"
    - If the question references previous answers (e.g., "tell me more about that"), look at the conversation history to understand what topic is being discussed
    - Then search the provided context for relevant information about that topic
    - If the context doesn't contain enough information, say "I cannot find sufficient information in the provided sources"
    - NEVER use outside knowledge or training data
    - You MUST cite sources with labels like [CITATION 1], [CITATION 2]
    - Every factual claim must have at least one citation
    - Only cite citations that actually exist in the provided context. NEVER invent or reference a citation number that was not given to you.
    - IMPORTANT: Try to reference ALL provided citations if they are relevant to the answer, even tangentially, but do not fabricate citations to reach a higher count.
    """
    conversation_history: str = dspy.InputField(desc='Previous questions and answers. Use this to understand what "this", "that", or "it" refers to in follow-up questions.')
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='Passages with citation labels like [CITATION 1], [CITATION 2]. Use ONLY these passages to answer.')
    response: str = dspy.OutputField(desc='Answer using ONLY the provided context. Include inline citations like [CITATION 1]. Only cite numbers that appear in the context.')
    citations: list[str] = dspy.OutputField(desc="List of citation sources used, e.g., [CITATION 1], [CITATION 2]. Only include citations that were actually present in the context and referenced in your answer.")


# Guaranteed slots per source in "both" mode (when that source has
# in-range results). With n_results=5 this yields at worst a 3/2 split.
MIN_PER_SOURCE = 2


class DualRAG(dspy.Module):
    """RAG over city council transcripts + news articles.

    Balancing: guaranteed-mix. Each source gets min(MIN_PER_SOURCE, n//2)
    reserved slots; leftover slots go to the best remaining candidates by
    distance regardless of source. Date filters are strict — no
    out-of-range supplementation.
    """

    def __init__(self, council_collection, articles_collection):
        self.council_collection = council_collection
        self.articles_collection = articles_collection
        self.respond = dspy.ChainOfThought(RAGQuestion)

    # ── retrieval helpers ─────────────────────────────────────────

    def _query_rows(self, collection, search_query, where, n, source_label):
        """One strict, date-filtered query. Returns [(dist, doc, id, meta, src)]."""
        if n <= 0 or collection is None:
            return []
        try:
            res = collection.query(
                query_texts=[search_query],
                n_results=n,
                where=where,
                include=['documents', 'metadatas', 'distances'],
            )
            return sorted(
                zip(res['distances'][0], res['documents'][0],
                    res['ids'][0], res['metadatas'][0],
                    [source_label] * len(res['ids'][0])),
                key=lambda r: r[0],
            )
        except BaseException as e:
            print(f"Error querying {source_label}: {e}")
            return []

    def _exact_rows(self, collection, search_query, where, exact_tokens, n, source_label):
        """Exact mode: large candidate pool, Python keyword filter, strict dates."""
        if n <= 0 or collection is None:
            return []

        def _doc_matches(doc):
            low = doc.lower()
            return all(t in low for t in exact_tokens)

        try:
            res = collection.query(
                query_texts=[search_query],
                n_results=max(n * 30, 300),
                where=where,                      # strict: never re-query without it
                include=['documents', 'metadatas', 'distances'],
            )
            rows = [
                (d, doc, id_, meta, source_label)
                for d, doc, id_, meta in zip(
                    res['distances'][0], res['documents'][0],
                    res['ids'][0], res['metadatas'][0])
                if _doc_matches(doc)
            ]
            rows.sort(key=lambda r: r[0])
            return rows[:n * 3]  # keep extra for blending; final cut happens later
        except BaseException as e:
            print(f"Exact match query failed ({source_label}): {e}")
            return []

    @staticmethod
    def _blend(council_rows, article_rows, n):
        """Guaranteed-mix selection: reserve slots per source, fill by distance."""
        reserved = min(MIN_PER_SOURCE, max(1, n // 2))
        picked = list(council_rows[:reserved]) + list(article_rows[:reserved])
        remaining = sorted(
            list(council_rows[reserved:]) + list(article_rows[reserved:]),
            key=lambda r: r[0],
        )
        picked += remaining[:max(0, n - len(picked))]
        picked.sort(key=lambda r: r[0])
        return picked[:n]

    # ── main entry ────────────────────────────────────────────────

    def forward(self, question, start_date, end_date, n_results=5,
                source_type="both", conversation_history="", search_mode="smart"):

        # Expand query for follow-up questions that use pronouns
        search_query = question
        if conversation_history and any(
            word in question.lower()
            for word in ['this', 'that', 'it', 'more', 'them', 'those']
        ):
            if "Previous Question:" in conversation_history:
                last_q = conversation_history.split("Previous Question:")[-1].split("\n")[0].strip()
                search_query = f"{last_q} {question}"

        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = (
            datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            + timedelta(days=1) - timedelta(seconds=1)
        )
        start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())

        council_where = {"$and": [{"date": {"$gte": start_ts}},
                                  {"date": {"$lte": end_ts}}]}
        articles_where = {"$and": [{"published_unix": {"$gte": start_ts}},
                                   {"published_unix": {"$lte": end_ts}}]}

        exact_tokens = (
            [w.lower() for w in re.findall(r'\b\w+\b', question)
             if w.lower() not in _STOP_WORDS and len(w) > 2]
            if search_mode == "exact" else []
        )

        want_council = source_type in ("both", "council")
        want_articles = source_type in ("both", "articles")

        if search_mode == "exact" and exact_tokens:
            council_rows = self._exact_rows(
                self.council_collection, search_query, council_where,
                exact_tokens, n_results if want_council else 0, 'council')
            article_rows = self._exact_rows(
                self.articles_collection, search_query, articles_where,
                exact_tokens, n_results if want_articles else 0, 'article')
        else:
            council_rows = self._query_rows(
                self.council_collection, search_query, council_where,
                n_results if want_council else 0, 'council')
            article_rows = self._query_rows(
                self.articles_collection, search_query, articles_where,
                n_results if want_articles else 0, 'article')

        if source_type == "both":
            rows = self._blend(council_rows, article_rows, n_results)
        else:
            rows = (council_rows or article_rows)[:n_results]

        if not rows:
            if source_type == "articles":
                msg = ("News article search returned no results in the selected "
                       "date range. Try widening the date range or mixed sources.")
            else:
                msg = ("I couldn't find any relevant information in the city council "
                       "transcripts or news articles within the selected date range. "
                       "Please try rephrasing or adjusting your date range.")
            return {'response': msg, 'context': '', 'ids': [], 'documents': [],
                    'meta': [], 'sources': [], 'citations': []}

        all_distances, all_documents, all_ids, all_meta, all_sources = (
            list(x) for x in zip(*rows)
        )

        context = '\n\n'.join(
            '### [CITATION %i] (Source: %s)\n%s' % (i, src.upper(), doc)
            for i, (doc, src) in enumerate(zip(all_documents, all_sources), start=1)
        )

        if search_mode == "exact":
            citations = ['[CITATION %d]' % i for i in range(1, len(all_documents) + 1)]
            return {
                'response': 'Exact match results for "%s":' % question,
                'citations': citations, 'context': context, 'ids': all_ids,
                'documents': all_documents, 'meta': all_meta, 'sources': all_sources,
            }

        response = self.respond(
            context=context, question=question,
            conversation_history=conversation_history,
        )
        response_text = response.response if hasattr(response, 'response') else str(response)
        citations = response.citations if hasattr(response, 'citations') else []
        if not citations:
            citations = list(dict.fromkeys(re.findall(r'\[CITATION \d+\]', response_text)))

        return {
            'response': response_text, 'citations': citations, 'context': context,
            'ids': all_ids, 'documents': all_documents, 'meta': all_meta,
            'sources': all_sources,
        }


# ── Citation HTML helpers ─────────────────────────────────────────

def citation2html_council(i, citation_no, row, start_time, quotes, names, summary):
    video_num = 'video%d' % citation_no
    return """
    <details>
      <summary><strong>Reference %d [CITY COUNCIL VIDEO]</strong></summary>
      <div style="padding:0.5em 1em;">
      <p>%s (%s)</p>
      <p>%s</p>
      <p><i>Quotes</i><br>%s</p>
      <p><i>Names:</i> %s </p>
      <video id="%s" width="640" height="360" controls preload="metadata">
        <source src="%s" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <script>
        document.getElementById('%s').addEventListener('loadedmetadata', () => {
          document.getElementById('%s').currentTime = %s;
        });
      </script>
      </div>
    </details>
    """ % (i, row.title, str(row.date)[:10],
           markdown.markdown(summary, extensions=["fenced_code", "tables"]),
           quotes, names, video_num, row.box_link,
           video_num, video_num, start_time)


def citation2html_article(i, meta, summary):
    meta = meta if isinstance(meta, dict) else {}
    published_display = ''

    published_unix = meta.get('published_unix')
    if published_unix is not None:
        try:
            published_display = datetime.fromtimestamp(
                int(published_unix), tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except Exception:
            published_display = ''

    if not published_display:
        published_raw = meta.get('published', '')
        if published_raw:
            try:
                published_display = parsedate_to_datetime(published_raw).strftime("%Y-%m-%d")
            except Exception:
                try:
                    published_display = datetime.fromisoformat(
                        published_raw.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                except Exception:
                    published_display = str(published_raw)

    return """
    <details>
      <summary><strong>Reference %d [NEWS ARTICLE]</strong></summary>
      <div style="padding:0.5em 1em;">
      <p><strong>%s</strong></p>
      <p><i>Source:</i> %s | <i>Published:</i> %s</p>
      <p>%s</p>
      <p><a href="%s" target="_blank">Read full article →</a></p>
      </div>
    </details>
    """ % (i,
           meta.get('title', 'Untitled'),
           meta.get('source', 'Unknown'),
           published_display,
           markdown.markdown(summary[:500] + '...' if len(summary) > 500 else summary),
           meta.get('url', '#'))


def format_citations(result):
    citations  = []
    cites_seen = set()

    if not result.get('documents'):
        return ''

    if not result.get('citations'):
        return (
            '<p style="color: red;"><strong>WARNING: No citations provided. '
            'This answer may not be grounded in the source documents.</strong></p>'
        )

    for i, c in enumerate(result['citations']):
        try:
            match = re.search(r'(\d+)', c)
            if not match:
                continue
            num = int(match.group(1)) - 1
        except (IndexError, ValueError):
            continue

        if num < 0 or num in cites_seen or num >= len(result.get('sources', [])):
            continue
        cites_seen.add(num)

        source_type = result['sources'][num]
        meta        = result['meta'][num]
        doc         = result['documents'][num]

        if source_type == 'council':
            mfile  = re.sub(r'\.summary$', '.mp4', os.path.basename(meta['file']))
            row = _council_meta.get(mfile)
            if row is None:
                print(f"Citation miss: '{mfile}' not in council metadata "
                      f"({len(_council_meta)} entries loaded). "
                      f"meta['file']={meta.get('file')!r}", flush=True)
                # Render a degraded citation rather than silently dropping it
                citations.append(
                    f'<details><summary><strong>Reference {i + 1} [CITY COUNCIL]</strong>'
                    f'</summary><div style="padding:0.5em 1em;">'
                    f'<p><em>Video metadata unavailable for {mfile}</em></p>'
                    f'<p>{doc[:400]}</p></div></details>'
                )
                continue
            quotes = '<ul>'
            for h in meta['quotes'].split('|||')[:3]:
                quotes += '\n<li>"%s"</li>' % h
            quotes += '\n</ul>\n'
            names = ', '.join(sorted(meta['names'].split('|||')))
            citations.append(
                citation2html_council(i + 1, num, row, meta['start_time'], quotes, names, doc)
            )
        elif source_type == 'article':
            citations.append(citation2html_article(i + 1, meta, doc))

    if not citations:
        return (
            '<p style="color: red;"><strong>WARNING: '
            'Citations could not be formatted.</strong></p>'
        )

    return '\n<br>\n'.join(citations)


# ── LM + embedding init ───────────────────────────────────────────
lm = dspy.LM(
    'openai/%s' % os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
    api_key='dummy',
    api_base=os.getenv('PROXY_BASE_URL'),
    extra_headers={'x-functions-key': os.getenv('FUNCTION_HOST_KEY', '')}
)
dspy.configure(lm=lm)

embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
)

def _load_collection(name: str, label: str):
    try:
        collection = chroma_client.get_collection(
            name=name, embedding_function=embed_fn
        )
        count = collection.count()
        print(f"Loaded {label} collection ({count} items)", flush=True)
        return collection
    except BaseException as e:
        print(f"{label.capitalize()} collection unavailable: {e}", flush=True)
        return None


council_collection = _load_collection("city_council", "city_council")
articles_collection = _load_collection("articles", "articles")

rag = DualRAG(council_collection, articles_collection)

# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'nocouncil-dev-secret-key')

# ── FIX 4: Bounded server-side history ───────────────────────────
# Prevents unbounded RAM growth on long-running or busy instances.
MAX_SESSIONS          = 200
MAX_TURNS_PER_SESSION = 20

_server_history: dict[str, list] = {}   # sid → list of turn dicts

def get_history() -> list:
    sid = session.get('sid')
    return _server_history.get(sid, []) if sid else []

def save_history(history: list) -> None:
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
    # Trim per-session
    _server_history[sid] = history[:MAX_TURNS_PER_SESSION]
    # Evict the oldest session when the cap is exceeded
    if len(_server_history) > MAX_SESSIONS:
        del _server_history[next(iter(_server_history))]


# ── HTML templates ────────────────────────────────────────────────

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CivicLens</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Oswald:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f6f5f0;
      --surface: #ffffff;
      --surface-alt: #f0eee8;
      --border: #e0ddd4;
      --border-light: #ebe8e0;
      --text: #2c2a26;
      --text-secondary: #6b6860;
      --text-muted: #9a968d;
      --accent: #1a4a8a;
      --accent-light: #e8eef8;
      --accent-hover: #123570;
      --gold: #c8a44e;
      --gold-light: #f7f2e4;
      --radius: 8px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Oswald", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
    }
    .top-bar {
      background: var(--accent);
      color: white;
      padding: 0.5em 2em;
      font-size: 0.75em;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .top-bar a { color: rgba(255,255,255,0.8); text-decoration: none; }
    .top-bar a:hover { color: white; }
    .container { max-width: 900px; margin: 0 auto; padding: 2em 1.5em 4em; }
    .header { text-align: center; margin-bottom: 2.5em; padding-top: 1em; }
    .header h1 {
      font-family: "Bebas Neue", sans-serif;
      font-weight: 800;
      font-size: 2.4em;
      color: var(--text);
      margin-bottom: 0.15em;
      letter-spacing: -0.02em;
    }
    .header .subtitle { color: var(--text-muted); font-size: 0.95em; }
    .header .gold-line {
      width: 60px; height: 3px;
      background: var(--gold);
      margin: 0.8em auto 0;
      border-radius: 2px;
    }
    .search-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.8em;
      box-shadow: var(--shadow-sm);
      margin-bottom: 2em;
    }
    .search-card textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.9em 1em;
      font-family: "Bebas Neue", sans-serif;
      font-size: 1em;
      resize: vertical;
      color: var(--text);
      background: var(--bg);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .search-card textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-light);
    }
    .search-card textarea::placeholder { color: var(--text-muted); }
    /* Segmented Smart / Exact toggle — visible up front, not reactive */
    .mode-toggle {
      display: inline-flex;
      margin-top: 1em;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px;
      gap: 2px;
    }
    .mode-pill {
      border: none;
      background: transparent;
      font-family: "Bebas Neue", sans-serif;
      font-size: 0.85em;
      letter-spacing: 0.03em;
      padding: 0.5em 1.1em;
      border-radius: 999px;
      color: var(--text-secondary);
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
    }
    .mode-pill.active { background: var(--accent); color: white; }
    .mode-pill:not(.active):hover { color: var(--text); }
    .filters-disclosure { margin-top: 1em; }
    .filters-disclosure summary {
      list-style: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35em;
      font-size: 0.85em;
      font-weight: 500;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .filters-disclosure summary::-webkit-details-marker { display: none; }
    .filters-disclosure summary::before { content: "\\25B8 "; font-size: 0.9em; color: var(--text-muted); }
    .filters-disclosure[open] summary::before { content: "\\25BE "; }
    .filters-body {
      display: flex; flex-wrap: wrap;
      gap: 0.8em; align-items: center;
      margin-top: 0.8em;
      padding-top: 0.9em;
      border-top: 1px solid var(--border-light);
    }
    .control-group { display: flex; align-items: center; gap: 0.4em; }
    .control-group label {
      font-size: 0.8em; font-weight: 500;
      color: var(--text-secondary);
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .control-group select,
    .control-group input[type="date"] {
      border: 1px solid var(--border);
      border-radius: 6px; padding: 0.45em 0.6em;
      font-family: "Bebas Neue", sans-serif;
      font-size: 0.88em; color: var(--text); background: var(--bg);
    }
    .hero-actions { display: flex; align-items: center; gap: 0.8em; margin-top: 1.2em; }
    .btn-ask {
      background: var(--accent); color: white;
      border: none; border-radius: 6px;
      padding: 0.6em 2em;
      font-family: "Bebas Neue", sans-serif;
      font-size: 0.95em; font-weight: 600;
      cursor: pointer; transition: background 0.2s, transform 0.1s;
      letter-spacing: 0.02em;
    }
    .btn-ask:hover { background: var(--accent-hover); }
    .btn-ask:active { transform: scale(0.98); }
    .btn-clear {
      background: none; border: 1px solid var(--border);
      border-radius: 6px; padding: 0.55em 1.2em;
      font-family: "Bebas Neue", sans-serif;
      font-size: 0.88em; color: var(--text-secondary);
      cursor: pointer; text-decoration: none;
      transition: background 0.2s, border-color 0.2s;
    }
    .btn-clear:hover { background: var(--surface-alt); border-color: var(--text-muted); }
    .loading-bar { display: none; margin: 1.5em 0; }
    .loading-bar.active { display: block; }
    .loading-bar p { margin: 0 0 0.6em; color: var(--text-muted); font-size: 0.9em; font-style: italic; }
    .progress-track { width: 100%; height: 4px; background: var(--border-light); border-radius: 2px; overflow: hidden; }
    .progress-fill {
      width: 0%; height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--gold));
      border-radius: 2px; animation: loading 2.5s ease-in-out infinite;
    }
    @keyframes loading { 0% { width: 5%; } 50% { width: 80%; } 100% { width: 95%; } }
    .results-header {
      font-family: "Bebas Neue", sans-serif; font-weight: 700;
      font-size: 1.5em; color: var(--text);
      margin-bottom: 1em; padding-bottom: 0.5em;
      border-bottom: 2px solid var(--gold);
      display: flex; align-items: baseline; gap: 0.6em;
    }
    .mode-badge {
      font-family: "Bebas Neue", sans-serif;
      font-size: 0.55em; font-weight: 600;
      padding: 0.25em 0.7em; border-radius: 4px;
      text-transform: uppercase; letter-spacing: 0.06em;
      vertical-align: middle;
      background: var(--surface-alt);
      color: var(--text-secondary);
      border: 1px solid var(--border);
    }
    .mode-badge.smart { background: var(--accent-light); color: var(--accent); border-color: #b8cce4; }
    .turn {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 1.5em;
      margin-bottom: 1.2em; box-shadow: var(--shadow-sm);
    }
    .turn:first-of-type { border-left: 3px solid var(--accent); box-shadow: var(--shadow-md); }
    .turn .q-label { font-weight: 600; color: var(--accent); font-size: 0.95em; margin-bottom: 0.5em; }
    .turn .a-label {
      font-weight: 600; color: var(--text-secondary);
      font-size: 0.82em; text-transform: uppercase;
      letter-spacing: 0.06em; margin: 0.8em 0 0.4em;
    }
    .turn .a-content { color: var(--text); line-height: 1.7; }
    .turn .a-content p { margin-bottom: 0.6em; }
    details {
      border: 1px solid var(--border-light); border-radius: 6px;
      margin-top: 0.8em; overflow: hidden;
    }
    details summary {
      padding: 0.7em 1em; cursor: pointer;
      font-size: 0.88em; font-weight: 600;
      color: var(--text-secondary); background: var(--surface-alt);
      transition: background 0.15s; list-style: none;
    }
    details summary::-webkit-details-marker { display: none; }
    details summary::before { content: "\\25B8 "; color: var(--gold); font-size: 0.9em; }
    details[open] summary::before { content: "\\25BE "; }
    details[open] summary { background: var(--surface-alt); }
    details > div { padding: 1em; }
    details video { border-radius: 6px; margin-top: 0.5em; }
    .footer {
      text-align: center; margin-top: 3em; padding-top: 1.5em;
      border-top: 1px solid var(--border-light);
      color: var(--text-muted); font-size: 0.78em;
    }
    @media (max-width: 600px) {
      .container { padding: 1em; }
      .header h1 { font-size: 1.8em; }
      .hero-actions { flex-direction: column; align-items: stretch; }
      .btn-ask { width: 100%; padding: 0.7em; }
    }
  </style>
</head>
<body>
  <div class="top-bar">
    <span>New Orleans Civic Search</span>
    <a href="/admin">Admin Panel</a>
  </div>
  <div class="container">
    <div class="header">
      <h1>CivicLens</h1>
      <div class="subtitle">Search city council meetings and civic news</div>
      <div class="gold-line"></div>
    </div>

    <div class="search-card">
      <form method="post" id="searchForm">
        <input type="hidden" name="search_mode" value="{{ search_mode }}">
        <textarea name="question" rows="3" placeholder="What would you like to know about New Orleans city government?">{{ question or "" }}</textarea>

        <div class="mode-toggle" id="modeToggle">
          <button type="button" class="mode-pill{{ ' active' if search_mode != 'exact' else '' }}" data-mode="smart">Smart Search</button>
          <button type="button" class="mode-pill{{ ' active' if search_mode == 'exact' else '' }}" data-mode="exact">Exact Match</button>
        </div>

        <details class="filters-disclosure">
          <summary>Filters</summary>
          <div class="filters-body">
            <div class="control-group">
              <label>Source</label>
              <select name="source_type">
                <option value="both"     {% if source_type == "both"     %}selected{% endif %}>Council &amp; News</option>
                <option value="council"  {% if source_type == "council"  %}selected{% endif %}>Council Only</option>
                <option value="articles" {% if source_type == "articles" %}selected{% endif %}>News Only</option>
              </select>
            </div>
            <div class="control-group">
              <label>References</label>
              <select name="n_results">
                {% for val in [5,10,15,20] %}
                  <option value="{{ val }}" {% if val == n_results %}selected{% endif %}>{{ val }}</option>
                {% endfor %}
              </select>
            </div>
            <div class="control-group">
              <label>From</label>
              <input type="date" name="start_date" value="{{ start_date or "" }}">
            </div>
            <div class="control-group">
              <label>To</label>
              <input type="date" name="end_date" value="{{ end_date or "" }}">
            </div>
          </div>
        </details>

        <div class="hero-actions">
          <button type="submit" class="btn-ask">Search</button>
          <a href="/clear" class="btn-clear">Clear</a>
        </div>
      </form>
    </div>

    <div class="loading-bar" id="loadingBar">
      <p id="loadingText">Searching...</p>
      <div class="progress-track"><div class="progress-fill"></div></div>
    </div>

    {% if history %}
      <div class="results-header">
        Results
        <span class="mode-badge{{ ' smart' if search_mode != 'exact' else '' }}">{{ "Exact Match" if search_mode == "exact" else "Smart Search" }}</span>
      </div>
      {% for turn in history %}
        <div class="turn">
          <div class="q-label">{{ turn.question }}</div>
          <div class="a-label">Answer</div>
          <div class="a-content">{{ turn.answer|safe }}</div>
        </div>
      {% endfor %}
    {% endif %}

    <div class="footer">CivicLens &middot; New Orleans City Council Public Records</div>
  </div>

  <script>
    document.querySelectorAll(".mode-pill").forEach(function(btn) {
      btn.addEventListener("click", function() {
        document.querySelectorAll(".mode-pill").forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        document.querySelector("#searchForm [name=\'search_mode\']").value = btn.dataset.mode;
      });
    });

    document.getElementById("searchForm").addEventListener("submit", function() {
      setTimeout(function() {
        document.getElementById("loadingBar").classList.add("active");
        var isExact = document.querySelector("#searchForm [name=\'search_mode\']").value === "exact";
        var msgs = isExact
          ? ["Scanning for exact matches...", "Filtering results...", "Preparing references..."]
          : ["Searching transcripts and articles...", "Reading through council meetings...", "Cross-referencing sources...", "Preparing your answer..."];
        var i = 0;
        setInterval(function() {
          i = Math.min(i + 1, msgs.length - 1);
          document.getElementById("loadingText").textContent = msgs[i];
        }, 3000);
      }, 100);
    });
  </script>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin — CivicLens</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Oswald:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f6f5f0; --surface: #ffffff; --surface-alt: #f0eee8;
      --border: #e0ddd4; --border-light: #ebe8e0;
      --text: #2c2a26; --text-secondary: #6b6860; --text-muted: #9a968d;
      --accent: #1a4a8a; --accent-light: #e8eef8; --accent-hover: #123570;
      --gold: #c8a44e; --gold-light: #f7f2e4;
      --radius: 8px; --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Oswald", sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }
    .top-bar { background: var(--accent); color: white; padding: 0.5em 2em; font-size: 0.75em; letter-spacing: 0.08em; text-transform: uppercase; display: flex; justify-content: space-between; align-items: center; }
    .top-bar a { color: rgba(255,255,255,0.8); text-decoration: none; }
    .top-bar a:hover { color: white; }
    .container { max-width: 900px; margin: 0 auto; padding: 2em 1.5em 4em; }
    .header { text-align: center; margin-bottom: 2em; padding-top: 1em; }
    .header h1 { font-family: "Bebas Neue", sans-serif; font-weight: 800; font-size: 2em; color: var(--text); margin-bottom: 0.15em; }
    .header .subtitle { color: var(--text-muted); font-size: 0.9em; }
    .header .gold-line { width: 60px; height: 3px; background: var(--gold); margin: 0.8em auto 0; border-radius: 2px; }
    .msg-success { padding: 0.8em 1.2em; margin-bottom: 1.5em; background: #e8eef8; border: 1px solid #b8cce4; border-left: 3px solid var(--accent); border-radius: var(--radius); font-size: 0.92em; color: var(--accent); font-weight: 500; }
    .msg-warn { padding: 0.8em 1.2em; margin-bottom: 1.5em; background: #fff8e6; border: 1px solid #e8d48a; border-left: 3px solid var(--gold); border-radius: var(--radius); font-size: 0.92em; color: #7a6520; }
    .section-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5em; margin-bottom: 1.2em; box-shadow: var(--shadow-sm); }
    .section-card h2 { font-family: "Bebas Neue", sans-serif; font-weight: 700; font-size: 1.2em; color: var(--text); margin-bottom: 0.2em; }
    .section-card .count { font-size: 0.8em; color: var(--text-muted); margin-bottom: 1em; }
    .form-row { display: flex; flex-wrap: wrap; gap: 0.8em; align-items: flex-end; }
    .form-field { display: flex; flex-direction: column; gap: 0.3em; }
    .form-field label { font-size: 0.78em; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em; }
    .form-field input[type="text"], .form-field select { border: 1px solid var(--border); border-radius: 6px; padding: 0.5em 0.7em; font-family: "Bebas Neue", sans-serif; font-size: 0.9em; color: var(--text); background: var(--bg); }
    .form-field input[type="text"]:focus, .form-field select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-light); }
    .btn-add { background: var(--accent); color: white; border: none; border-radius: 6px; padding: 0.55em 1.4em; font-family: "Bebas Neue", sans-serif; font-size: 0.88em; font-weight: 600; cursor: pointer; transition: background 0.2s; white-space: nowrap; }
    .btn-add:hover { background: var(--accent-hover); }
    details { border: 1px solid var(--border-light); border-radius: 6px; margin-top: 1em; overflow: hidden; }
    details summary { padding: 0.7em 1em; cursor: pointer; font-size: 0.85em; font-weight: 600; color: var(--text-secondary); background: var(--surface-alt); list-style: none; }
    details summary::-webkit-details-marker { display: none; }
    details summary::before { content: "\\25B8 "; color: var(--gold); }
    details[open] summary::before { content: "\\25BE "; }
    details[open] summary { background: var(--surface-alt); }
    details ul { padding: 0.8em 1.2em 0.8em 2em; max-height: 300px; overflow-y: auto; }
    details li { padding: 0.3em 0; font-size: 0.88em; color: var(--text-secondary); border-bottom: 1px solid var(--border-light); }
    details li:last-child { border-bottom: none; }
    .arrow { color: var(--gold); font-weight: 600; }
    .footer { text-align: center; margin-top: 2em; padding-top: 1.5em; border-top: 1px solid var(--border-light); color: var(--text-muted); font-size: 0.78em; }
  </style>
</head>
<body>
  <div class="top-bar">
    <a href="/">&larr; Back to Search</a>
    <span>Admin Panel</span>
  </div>
  <div class="container">
    <div class="header">
      <h1>Dictionary Management</h1>
      <div class="subtitle">Manage correction dictionaries for the transcript pipeline</div>
      <div class="gold-line"></div>
    </div>

    {% if not corrections_available %}
      <div class="msg-warn">correction.py not found — admin features are unavailable in this environment.</div>
    {% else %}

    {% if message %}
      <div class="msg-success">{{ message }}</div>
    {% endif %}

    <div class="section-card">
      <h2>Add a Name</h2>
      <div class="count">Currently: {{ n_first }} first names, {{ n_last }} last names</div>
      <form method="post">
        <input type="hidden" name="action" value="add_name">
        <div class="form-row">
          <div class="form-field">
            <label>Name</label>
            <input type="text" name="name_value" placeholder="e.g. Moreno" required>
          </div>
          <div class="form-field">
            <label>Type</label>
            <select name="name_type">
              <option value="first">First Name</option>
              <option value="last">Last Name</option>
            </select>
          </div>
          <button type="submit" class="btn-add">Add Name</button>
        </div>
      </form>
    </div>

    <div class="section-card">
      <h2>Add a Street</h2>
      <div class="count">Currently: {{ n_streets }} streets</div>
      <form method="post">
        <input type="hidden" name="action" value="add_street">
        <div class="form-row">
          <div class="form-field" style="flex:1;">
            <label>Street Name</label>
            <input type="text" name="street_value" placeholder="e.g. Tchoupitoulas Street" required style="width:100%;">
          </div>
          <button type="submit" class="btn-add">Add Street</button>
        </div>
      </form>
    </div>

    <div class="section-card">
      <h2>Add a Hardcoded Correction</h2>
      <div class="count">Currently: {{ n_hardcoded }} hardcoded corrections</div>
      <form method="post">
        <input type="hidden" name="action" value="add_hardcoded">
        <div class="form-row">
          <div class="form-field">
            <label>Misspelling</label>
            <input type="text" name="misspelling" placeholder="e.g. Geruso" required>
          </div>
          <div class="form-field">
            <label>Correct Spelling</label>
            <input type="text" name="correct_spelling" placeholder="e.g. Giarrusso" required>
          </div>
          <button type="submit" class="btn-add">Add Correction</button>
        </div>
      </form>
      {% if hardcoded_list %}
      <details>
        <summary>View all hardcoded corrections ({{ n_hardcoded }})</summary>
        <ul>
        {% for k, v in hardcoded_list %}
          <li>"{{ k }}" <span class="arrow">&rarr;</span> "{{ v }}"</li>
        {% endfor %}
        </ul>
      </details>
      {% endif %}
    </div>

    {% endif %}
    <div class="footer">CivicLens &middot; Admin Panel</div>
  </div>
</body>
</html>
'''


# ── Routes ────────────────────────────────────────────────────────

@app.route("/clear")
def clear():
    sid = session.get('sid')
    if sid and sid in _server_history:
        del _server_history[sid]
    return redirect("/")


@app.route("/", methods=["GET", "POST"])
def index():
    question    = None
    n_results   = 5
    start_date  = default_start
    end_date    = default_end
    source_type = "both"
    search_mode = "smart"
    history     = get_history()

    if request.method == "POST":
        question    = request.form.get("question", "").strip()
        start_date  = request.form.get("start_date") or default_start
        end_date    = request.form.get("end_date")   or default_end
        source_type = request.form.get("source_type", "both")
        search_mode = request.form.get("search_mode", "smart")
        try:
            n_results = int(request.form.get("n_results", 5))
        except ValueError:
            n_results = 5

        if question:
            history_str = ""
            for turn in history[-3:]:
                history_str += f"Previous Question: {turn['question']}\n"
                history_str += f"Previous Answer: {turn['answer_text']}\n\n"

            result = rag(
                question=question,
                n_results=n_results,
                start_date=start_date,
                end_date=end_date,
                source_type=source_type,
                search_mode=search_mode,
                conversation_history=history_str
            )

            answer_html = result.get('response', '') + '<br><br>\n' + format_citations(result)
            history = [{'question': question, 'answer': answer_html,
                        'answer_text': result.get('response', '')}] + history
            save_history(history)
            question = None

    return render_template_string(
        HTML_TEMPLATE,
        question=question,
        history=history,
        n_results=n_results,
        start_date=start_date,
        end_date=end_date,
        source_type=source_type,
        search_mode=search_mode
    )


# ── Admin ─────────────────────────────────────────────────────────

_app_dir     = os.path.dirname(os.path.abspath(__file__))
_admin_dicts: dict = {}
if _corrections_available:
    try:
        _admin_dicts = load_dictionaries(
            english_path=os.path.join(_app_dir, 'english_words.json'),
            names_path=os.path.join(_app_dir, 'nola_names.json'),
            streets_path=os.path.join(_app_dir, 'nola_streets.json'),
        )
    except Exception as e:
        print(f'Warning: Could not load admin dictionaries: {e}')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    global _admin_dicts
    message  = None
    app_dir  = os.path.dirname(os.path.abspath(__file__))

    if not _corrections_available:
        return render_template_string(
            ADMIN_TEMPLATE, corrections_available=False,
            message=None, n_first=0, n_last=0,
            n_streets=0, n_hardcoded=0, hardcoded_list=[]
        )

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_name':
            name_value = request.form.get('name_value', '').strip()
            name_type  = request.form.get('name_type', 'last')
            if name_value:
                names_path = os.path.join(app_dir, 'nola_names.json')
                with open(names_path, 'r', encoding='utf-8') as f:
                    names_data = json.load(f)
                key = 'first_names' if name_type == 'first' else 'last_names'
                if name_value not in names_data.get(key, []):
                    names_data.setdefault(key, []).append(name_value)
                    with open(names_path, 'w', encoding='utf-8') as f:
                        json.dump(names_data, f, indent=2, ensure_ascii=False)
                    _admin_dicts = load_dictionaries(
                        english_path=os.path.join(app_dir, 'english_words.json'),
                        names_path=names_path,
                        streets_path=os.path.join(app_dir, 'nola_streets.json'),
                    )
                    message = f'Added {name_type} name: {name_value}'
                else:
                    message = f"'{name_value}' already exists in {key}"

        elif action == 'add_street':
            street_value = request.form.get('street_value', '').strip()
            if street_value:
                streets_path = os.path.join(app_dir, 'nola_streets.json')
                with open(streets_path, 'r', encoding='utf-8') as f:
                    streets_data = json.load(f)
                if street_value not in streets_data:
                    streets_data.append(street_value)
                    with open(streets_path, 'w', encoding='utf-8') as f:
                        json.dump(streets_data, f, indent=2, ensure_ascii=False)
                    _admin_dicts = load_dictionaries(
                        english_path=os.path.join(app_dir, 'english_words.json'),
                        names_path=os.path.join(app_dir, 'nola_names.json'),
                        streets_path=streets_path,
                    )
                    message = f'Added street: {street_value}'
                else:
                    message = f"'{street_value}' already exists in streets"

        elif action == 'add_hardcoded':
            misspelling      = request.form.get('misspelling', '').strip()
            correct_spelling = request.form.get('correct_spelling', '').strip()
            if misspelling and correct_spelling:
                hc_path = os.path.join(app_dir, 'hardcoded_corrections.json')
                try:
                    with open(hc_path, 'r', encoding='utf-8') as f:
                        hc_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    hc_data = {}
                hc_data[misspelling.lower()] = correct_spelling
                with open(hc_path, 'w', encoding='utf-8') as f:
                    json.dump(hc_data, f, indent=2, ensure_ascii=False)
                load_hardcoded_corrections(hc_path)
                message = f"Added correction: '{misspelling}' → '{correct_spelling}'"

    from correction import HARDCODED_CORRECTIONS
    names_path = os.path.join(app_dir, 'nola_names.json')
    try:
        with open(names_path, 'r', encoding='utf-8') as f:
            nd = json.load(f)
        n_first = len(nd.get('first_names', []))
        n_last  = len(nd.get('last_names',  []))
    except Exception:
        n_first = n_last = 0

    return render_template_string(
        ADMIN_TEMPLATE,
        corrections_available=True,
        message=message,
        n_first=n_first,
        n_last=n_last,
        n_streets=len(_admin_dicts.get('streets', [])),
        n_hardcoded=len(HARDCODED_CORRECTIONS),
        hardcoded_list=sorted(HARDCODED_CORRECTIONS.items()),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
