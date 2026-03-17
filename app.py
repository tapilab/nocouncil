"""
Simple Flask app to serve a RAG model over video transcripts AND news articles.
- Single ChromaDB with both city_council and nola_articles collections
- Multi-turn conversation support
- Requires OPENAI_API_KEY
- model set by OPENAI_MODEL (default: gpt-4o-mini)
"""
import os
from datetime import datetime
import dspy
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
import secrets
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
import json
import markdown
import openai
import pandas as pd
import re

# Load environment variables
load_dotenv()

# Correction imports for admin page (dictionary management)
try:
    from nocouncil.correction import load_dictionaries, load_hardcoded_corrections
    _corrections_available = True
except ImportError:
    _corrections_available = False

# Single ChromaDB client for both collections
chroma_client = PersistentClient(
    path=os.getenv('CHROMA_DB_DIR', '/models/chroma_db'),
    settings=Settings(anonymized_telemetry=False)
)

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
    - IMPORTANT: Try to reference ALL provided citations if they are relevant to the answer, even tangentially
    """
    conversation_history: str = dspy.InputField(desc='Previous questions and answers. Use this to understand what "this", "that", or "it" refers to in follow-up questions.')
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='Passages with citation labels like [CITATION 1], [CITATION 2]. Use ONLY these passages to answer. Try to incorporate information from as many citations as possible.')
    response: str = dspy.OutputField(desc='Answer using ONLY the provided context. Include inline citations like [CITATION 1]. Reference multiple sources when possible.')
    citations: list[str] = dspy.OutputField(desc="List of ALL citation sources used, e.g., [CITATION 1], [CITATION 2]. Must include every source referenced in your answer. Aim to use most or all provided citations.")

def filename2date(filename):
    mp4 = re.sub('.summary', '.mp4', filename.split('/')[-1])
    return df[df.video.str.contains(mp4)].iloc[0].date

class DualRAG(dspy.Module):
    """RAG that searches both city council transcripts and news articles"""

    def __init__(self, council_collection, articles_collection):
        self.council_collection = council_collection
        self.articles_collection = articles_collection
        self.respond = dspy.ChainOfThought(RAGQuestion)

    def forward(self, question, start_date, end_date, n_results=5, source_type="both", conversation_history=""):
        # Expand query if it's a follow-up question
        search_query = question
        if conversation_history and any(word in question.lower() for word in ['this', 'that', 'it', 'more', 'them', 'those']):
            if "Previous Question:" in conversation_history:
                last_q = conversation_history.split("Previous Question:")[-1].split("\n")[0].strip()
                search_query = f"{last_q} {question}"

        start_dt = datetime.fromisoformat(start_date)
        end_dt   = datetime.fromisoformat(end_date)
        start_ts = int(start_dt.timestamp())
        end_ts   = int(end_dt.timestamp())

        where = {
          "$and": [
            {"date": {"$gte": start_ts}},
            {"date": {"$lte": end_ts}}
          ]
        }

        all_documents = []
        all_ids = []
        all_meta = []
        all_sources = []
        all_distances = []

        if source_type == "both":
            council_n = n_results * 2
            articles_n = n_results * 2
        elif source_type == "council":
            council_n = n_results
            articles_n = 0
        else:
            council_n = 0
            articles_n = n_results

        if council_n > 0 and self.council_collection is not None:
            try:
                council_results = self.council_collection.query(
                    query_texts=[search_query],
                    n_results=council_n,
                    where=where,
                    include=['documents', 'metadatas', 'distances']
                )
                all_documents.extend(council_results['documents'][0])
                all_ids.extend(council_results['ids'][0])
                all_meta.extend(council_results['metadatas'][0])
                all_sources.extend(['council'] * len(council_results['documents'][0]))
                all_distances.extend(council_results['distances'][0])
            except Exception as e:
                print(f"Error querying council: {e}")

        if articles_n > 0 and self.articles_collection is not None:
            try:
                article_results = self.articles_collection.query(
                    query_texts=[search_query],
                    n_results=articles_n,
                    where=where,
                    include=['documents', 'metadatas', 'distances']
                )
                if len(article_results['documents'][0]) == 0:
                    article_results = self.articles_collection.query(
                        query_texts=[search_query],
                        n_results=articles_n,
                        include=['documents', 'metadatas', 'distances']
                    )
                all_documents.extend(article_results['documents'][0])
                all_ids.extend(article_results['ids'][0])
                all_meta.extend(article_results['metadatas'][0])
                all_sources.extend(['article'] * len(article_results['documents'][0]))
                all_distances.extend(article_results['distances'][0])
            except Exception as e:
                print(f"Error querying articles: {e}")

        if source_type == "both" and len(all_documents) > 0:
            if source_type == "both":
                # Split evenly: half council, half articles
                council_items = [(d,doc,i,m,s) for d,doc,i,m,s in zip(all_distances, all_documents, all_ids, all_meta, all_sources) if s == 'council']
                article_items = [(d,doc,i,m,s) for d,doc,i,m,s in zip(all_distances, all_documents, all_ids, all_meta, all_sources) if s == 'article']
                council_items.sort(key=lambda x: x[0])
                article_items.sort(key=lambda x: x[0])
                half = n_results // 2
                council_take = half + (n_results % 2)  # council gets the extra if odd
                article_take = half
                combined = council_items[:council_take] + article_items[:article_take]
                combined.sort(key=lambda x: x[0])
            else:
                combined = list(zip(all_distances, all_documents, all_ids, all_meta, all_sources))
                combined.sort(key=lambda x: x[0])
                combined = combined[:n_results]
            if combined:
                all_distances, all_documents, all_ids, all_meta, all_sources = zip(*combined)
                all_documents = list(all_documents)
                all_ids = list(all_ids)
                all_meta = list(all_meta)
                all_sources = list(all_sources)
                all_distances = list(all_distances)

        if not all_documents:
            return {
                'response': "I couldn't find any relevant information in the city council transcripts or news articles to answer your question. Please try rephrasing or adjusting your date range.",
                'context': '',
                'ids': [],
                'documents': [],
                'meta': [],
                'sources': [],
                'citations': []
            }

        context = '\n\n'.join(
            '### [CITATION %i] (Source: %s)\n%s' % (i, src.upper(), doc)
            for i, (doc, src) in enumerate(zip(all_documents, all_sources))
        )

        response = self.respond(
            context=context,
            question=question,
            conversation_history=conversation_history
        )

        return {
            'response': response.response if hasattr(response, 'response') else str(response),
            'citations': response.citations if hasattr(response, 'citations') else [],
            'context': context,
            'ids': all_ids,
            'documents': all_documents,
            'meta': all_meta,
            'sources': all_sources
        }

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
           quotes, names, video_num, row.box_link, video_num, video_num, start_time)

def citation2html_article(i, meta, summary):
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
           meta.get('published', '')[:16] if meta.get('published') else '',
           markdown.markdown(summary[:500] + '...' if len(summary) > 500 else summary),
           meta.get('url', '#'))

def format_citations(result):
    citations = []
    cites_seen = set()

    if not result.get('citations') or len(result.get('citations', [])) == 0:
        return '<p style="color: red;"><strong>WARNING: No citations provided. This answer may not be grounded in the source documents.</strong></p>'

    for i, c in enumerate(result.get('citations', [])):
        try:
            match = re.search(r'(\d+)', c)
            if not match:
                continue
            num = int(match.group(1))
        except (IndexError, ValueError):
            continue

        if num in cites_seen:
            continue
        cites_seen.add(num)

        if num >= len(result.get('sources', [])):
            continue

        source_type = result['sources'][num]
        meta = result['meta'][num]
        doc = result['documents'][num]

        if source_type == 'council':
            mfile = re.sub('.summary', '.mp4', meta['file'].split('/')[-1])
            quotes = '<ul>'
            for h in meta['quotes'].split('|||')[:3]:
                quotes += '\n<li>"%s"</li>' % h
            quotes += '\n</ul>\n'
            names = ', '.join(sorted(meta['names'].split('|||')))
            row = df[df.video.str.contains(mfile)].iloc[0]
            citations.append(citation2html_council(i+1, num, row, meta['start_time'], quotes, names, doc))
        elif source_type == 'article':
            citations.append(citation2html_article(i+1, meta, doc))

    if len(citations) == 0:
        return '<p style="color: red;"><strong>WARNING: Citations could not be formatted.</strong></p>'

    return '\n<br>\n'.join(citations)


# ── LM init with Azure proxy ─────────────────────────────────────
lm = dspy.LM('openai/%s' % os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
             api_key='dummy',
             api_base=os.getenv('PROXY_BASE_URL'),
             extra_headers={'x-functions-key': os.getenv('FUNCTION_HOST_KEY', '')})
dspy.configure(lm=lm)

embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
)

# Load city council collection
try:
    council_collection = chroma_client.get_collection(
        name="city_council",
        embedding_function=embed_fn
    )
    print(f"Loaded city_council collection ({council_collection.count()} documents)")
except Exception as e:
    print(f"City council collection not found: {e}")
    council_collection = None

# Load articles collection (optional — falls back gracefully if missing)
try:
    articles_collection = chroma_client.get_collection(
        name="articles",
        embedding_function=embed_fn
    )
    print(f"Loaded articles collection ({articles_collection.count()} articles)")
except Exception as e:
    print(f"Articles collection not found (council-only mode): {e}")
    articles_collection = None

# Load council metadata
try:
    df = pd.read_json(os.environ.get("FLY_DATA", './') + '/data.jsonl', lines=True)
    default_start = df.date.min().strftime("%Y-%m-%d")
    default_end   = df.date.max().strftime("%Y-%m-%d")
except Exception as e:
    print(f"Council metadata not found: {e}")
    df = None
    default_start = "2020-01-01"
    default_end = datetime.now().strftime("%Y-%m-%d")

rag = DualRAG(council_collection, articles_collection)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'nocouncil-dev-secret-key')

# Server-side history storage (avoids 4KB cookie limit)
_server_history = {}

def get_history():
    sid = session.get('sid')
    if not sid:
        return []
    return _server_history.get(sid, [])

def save_history(history):
    sid = session.get('sid')
    if not sid:
        import uuid
        sid = str(uuid.uuid4())
        session['sid'] = sid
    _server_history[sid] = history

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ask a Question</title>
  <style>
    body { font-family: sans-serif; max-width: 860px; margin: 2em auto; padding: 0 1em; }
    textarea { width: 100%; box-sizing: border-box; }
    .controls { display: flex; flex-wrap: wrap; gap: 1em; align-items: center; margin: 0.5em 0; }
    button { padding: 0.4em 1.2em; cursor: pointer; }
    .clear-btn { background: #eee; border: 1px solid #ccc; }
    .history-item { border: 1px solid #ddd; border-radius: 6px; padding: 1em; margin-top: 1.5em; background: #fafafa; }
    .history-item.latest { background: #fff; border-color: #aaa; }
    .q-label { font-weight: bold; color: #333; margin-bottom: 0.3em; }
    .a-label { font-weight: bold; color: #555; margin: 0.8em 0 0.3em; }
    hr { border: none; border-top: 1px solid #eee; margin: 2em 0; }
  </style>
</head>
<body>
  <h1>Ask a Question</h1>
  <p style="font-size:0.9em;"><a href="/admin">Admin</a></p>
  <form method="post">
    <textarea name="question" rows="3" placeholder="Ask a follow-up or a new question...">{{ question or '' }}</textarea>
    <div class="controls">
      <label>Search in:
        <select name="source_type">
          <option value="both" {% if source_type == "both" %}selected{% endif %}>Both Council & News</option>
          <option value="council" {% if source_type == "council" %}selected{% endif %}>City Council Only</option>
          <option value="articles" {% if source_type == "articles" %}selected{% endif %}>News Articles Only</option>
        </select>
      </label>
      <label>References:
        <select name="n_results">
          {% for val in [5,10,15,20] %}
            <option value="{{ val }}" {% if val == n_results %}selected{% endif %}>{{ val }}</option>
          {% endfor %}
        </select>
      </label>
      <label>Start: <input type="date" name="start_date" value="{{ start_date or '' }}"></label>
      <label>End: <input type="date" name="end_date" value="{{ end_date or '' }}"></label>
      <button type="submit">Ask</button>
      <a href="/clear"><button type="button" class="clear-btn">Clear conversation</button></a>
    </div>
  </form>

  {% if history %}
    <hr>
    <h2>Conversation</h2>
    {% for turn in history %}
      <div class="history-item {% if loop.first %}latest{% endif %}">
        <div class="q-label">Q: {{ turn.question }}</div>
        <div class="a-label">A:</div>
        <div>{{ turn.answer|safe }}</div>
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
'''

@app.route("/clear")
def clear():
    sid = session.get('sid')
    if sid and sid in _server_history:
        del _server_history[sid]
    return redirect("/")

@app.route("/", methods=["GET", "POST"])
def index():
    question = None
    n_results = 5
    start_date = default_start
    end_date = default_end
    source_type = "both"

    history = get_history()

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        start_date = request.form.get("start_date") or default_start
        end_date = request.form.get("end_date") or default_end
        source_type = request.form.get("source_type", "both")
        try:
            n_results = int(request.form.get("n_results", 5))
        except ValueError:
            n_results = 5

        if question:
            # Build history string for LM (last 3 turns)
            history_str = ""
            if history:
                for turn in history[-3:]:
                    history_str += f"Previous Question: {turn['question']}\n"
                    history_str += f"Previous Answer: {turn['answer_text']}\n\n"

            result = rag(
                question=question,
                n_results=n_results,
                start_date=start_date,
                end_date=end_date,
                source_type=source_type,
                conversation_history=history_str
            )

            answer_html = result.get('response', '') + '<br><br>\n' + format_citations(result)

            new_turn = {
                'question': question,
                'answer': answer_html,
                'answer_text': result.get('response', '')
            }
            history = [new_turn] + history
            save_history(history)
            question = None

    return render_template_string(HTML_TEMPLATE,
        question=question,
        history=history,
        n_results=n_results,
        start_date=start_date,
        end_date=end_date,
        source_type=source_type)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


# ═══════════════════════════════════════════════════════════════════
# Admin page — manage correction dictionaries (names, streets, etc.)
# ═══════════════════════════════════════════════════════════════════

ADMIN_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin — Dictionary Management</title>
  <style>body { font-family: sans-serif; max-width: 860px; margin: 2em auto; padding: 0 1em; }</style>
</head>
<body>
  <h1>Admin — Dictionary Management</h1>
  <p><a href="/">← Back to search</a></p>

  {% if not corrections_available %}
    <div style="padding:0.5em 1em; background:#fff3cd; border:1px solid #ffc107;">
      correction.py not found — admin features unavailable in this environment.
    </div>
  {% else %}

  {% if message %}
    <div style="padding:0.5em 1em; margin:1em 0; background:#d4edda; border:1px solid #c3e6cb;">
      {{ message }}
    </div>
  {% endif %}

  <hr>
  <h2>Add a Name</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_name">
    <label>Name: <input type="text" name="name_value" size="30" required></label>
    &nbsp;
    <label>Type:
      <select name="name_type">
        <option value="first">First Name</option>
        <option value="last">Last Name</option>
      </select>
    </label>
    &nbsp;
    <button type="submit">Add Name</button>
  </form>
  <p><small>Currently: {{ n_first }} first names, {{ n_last }} last names</small></p>

  <hr>
  <h2>Add a Street</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_street">
    <label>Street name (e.g. "Tchoupitoulas Street"):
      <input type="text" name="street_value" size="50" required>
    </label>
    &nbsp;
    <button type="submit">Add Street</button>
  </form>
  <p><small>Currently: {{ n_streets }} streets</small></p>

  <hr>
  <h2>Add a Hardcoded Correction</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_hardcoded">
    <label>Misspelling: <input type="text" name="misspelling" size="25" required></label>
    &nbsp;
    <label>Correct spelling: <input type="text" name="correct_spelling" size="25" required></label>
    &nbsp;
    <button type="submit">Add Correction</button>
  </form>
  <p><small>Currently: {{ n_hardcoded }} hardcoded corrections</small></p>

  {% if hardcoded_list %}
  <details>
    <summary>View all hardcoded corrections</summary>
    <ul>
    {% for k, v in hardcoded_list %}
      <li>"{{ k }}" → "{{ v }}"</li>
    {% endfor %}
    </ul>
  </details>
  {% endif %}

  {% endif %}
</body>
</html>
'''


# Load admin dictionaries at startup if correction module is available
_app_dir = os.path.dirname(os.path.abspath(__file__))
_admin_dicts = {}
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
    message = None
    app_dir = os.path.dirname(os.path.abspath(__file__))

    if not _corrections_available:
        return render_template_string(ADMIN_TEMPLATE, corrections_available=False,
                                      message=None, n_first=0, n_last=0,
                                      n_streets=0, n_hardcoded=0, hardcoded_list=[])

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_name':
            name_value = request.form.get('name_value', '').strip()
            name_type = request.form.get('name_type', 'last')
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
            misspelling = request.form.get('misspelling', '').strip()
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

    from nocouncil.correction import HARDCODED_CORRECTIONS
    names_path = os.path.join(app_dir, 'nola_names.json')
    try:
        with open(names_path, 'r', encoding='utf-8') as f:
            nd = json.load(f)
        n_first = len(nd.get('first_names', []))
        n_last = len(nd.get('last_names', []))
    except Exception:
        n_first = n_last = 0

    hardcoded_list = sorted(HARDCODED_CORRECTIONS.items())
    return render_template_string(
        ADMIN_TEMPLATE,
        corrections_available=True,
        message=message,
        n_first=n_first,
        n_last=n_last,
        n_streets=len(_admin_dicts.get('streets', [])),
        n_hardcoded=len(HARDCODED_CORRECTIONS),
        hardcoded_list=hardcoded_list,
    )
