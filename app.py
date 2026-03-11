"""
Simple Flask app to serve a RAG model over video transcripts.
- Reads chroma vector database from CHROMA_DB_DIR
- Requires OPENAI_API_KEY (via OPEN_AI_KEY env var)
- model set by OPENAI_MODEL (default: gpt-4o-mini)
- port set by PORT (8080 default)

Transcript correction is handled upstream in the ETL pipeline
(correct_transcript.py), so app.py serves already-corrected data.

The /admin page allows managing correction dictionaries (names,
streets, hardcoded corrections) which are used by the ETL pipeline.

NB: By default, embeds queries using all-MiniLM-L6-v2. The chroma db will
need to use the same.
"""
import os
from datetime import datetime
import dspy
from flask import Flask, request, jsonify, render_template_string
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
import json
import markdown
import multiprocessing
import openai
import os
import pandas as pd
import re

# ── Correction imports (for admin page dictionary management only) ──
from correction import load_dictionaries, load_hardcoded_corrections

# load_dotenv()

# ═════════════════════════════════════════════════════════════════
# ChromaDB client
# ═════════════════════════════════════════════════════════════════
chroma_client = PersistentClient(
    path=os.getenv('CHROMA_DB_DIR', '/models/chroma_db'),
    settings=Settings(anonymized_telemetry=False)
)


# ═════════════════════════════════════════════════════════════════
# DSPy RAG signature & module
# ═════════════════════════════════════════════════════════════════
class RAGQuestion(dspy.Signature):
    """
    Answer this question about New Orleans City Council meetings based only on the provided context.
    Cite as many relevant sources as you can with citation labels like [CITATION 1].
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='Passages with citation labels like [CITATION 1], [CITATION 2].')
    response: str = dspy.OutputField(desc='Answer without inline citations.')
    citations: list[str] = dspy.OutputField(desc="List of citation sources used, e.g., [[CITATION 1], [CITATION 2]]. Cite as many as are relevant.")


def filename2date(filename):
    mp4 = re.sub('.summary', '.mp4', filename.split('/')[-1])
    return df[df.video.str.contains(mp4)].iloc[0].date


class RAG(dspy.Module):

    def __init__(self, collection):
        self.collection = collection
        self.respond = dspy.ChainOfThought(RAGQuestion)

    def forward(self, question, start_date, end_date, n_results=5):
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
        result = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            where=where
        )

        context = result['documents'][0]
        context_str = '\n\n'.join(
            '### [CITATION %i]\n%s' % (i, s) for i, s in enumerate(context)
        )

        response = self.respond(context=context_str, question=question)
        response['context'] = context_str
        response['ids'] = result['ids'][0]
        response['documents'] = result['documents'][0]
        response['meta'] = result['metadatas'][0]
        return response


# ═════════════════════════════════════════════════════════════════
# Helper functions (unchanged from original)
# ═════════════════════════════════════════════════════════════════

def get_text_by_ids(js, start, end, no_speech_thresh=.2):
    txts = []
    for j in js:
        if j['no_speech_prob'] < no_speech_thresh and j['id'] >= start and j['id'] <= end:
            txts.append(j['text'])
    return ' '.join(txts)


def get_text(js, start, end, no_speech_thresh=.2):
    txts = []
    for j in js[start:end]:
        if j['no_speech_prob'] < no_speech_thresh:
            txts.append(j['text'])
    return ' '.join(txts)


def embed_video(video_id, video_url, seek):
    return """
        <video id="%s" width="640" height="360" controls>
          <source src="%s" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <script>
          const %s = document.getElementById("%s");
          %s.addEventListener('loadedmetadata', function () {
            %s.currentTime = %d;
          });
        </script>
    """ % (video_id, video_url, video_id, video_id, video_id, video_id, int(seek))


def citation2html(i, citation_no, row, start_time, quotes, names, summary):
    video_num = 'video%d' % citation_no
    return """
    <details>
      <summary><strong>Reference %d</strong></summary>
      <div style="padding:0.5em 1em;">
      <p>%s (%s)</p>
      <p>%s</p>
      <p><i>Quotes</i><br>
          %s
      </p>
      <p><i>Names:</i> %s </p>
      <video id="%s" width="640" height="360" controls preload="metadata">
        <source
          src="%s"
          type="video/mp4">
          Your browser does not support the video tag.
      </video>
      <script>
        document.getElementById('%s').addEventListener('loadedmetadata', () => {
          document.getElementById('%s').currentTime = %s;
        });
      </script>
      </div>
    </details>
    """ % (
        i,
        row.title, str(row.date)[:10],
        markdown.markdown(summary, extensions=["fenced_code", "tables"]),
        quotes, names, video_num, row.box_link, video_num, video_num,
        start_time
    )


def format_citations(result):
    citations = []
    cites_seen = set()
    for i, c in enumerate(result.citations):
        num = int(re.findall(r'([\d+])', c)[0])
        if num in cites_seen:
            continue
        cites_seen.add(num)
        meta = result['meta'][num]
        mfile = re.sub('.summary', '.mp4', meta['file'].split('/')[-1])
        quotes = '<ul>'
        for h in meta['quotes'].split('|||')[:3]:
            quotes += '\n<li>"%s"</li>' % h
        quotes += '\n</ul>\n'
        names = ', '.join(sorted(meta['names'].split('|||')))
        row = df[df.video.str.contains(mfile)].iloc[0]
        citations.append(
            citation2html(i+1, num, row, meta['start_time'], quotes, names, result['documents'][num])
        )
    return '\n<br>\n'.join(citations)


# ═════════════════════════════════════════════════════════════════
# Global initialisation
# ═════════════════════════════════════════════════════════════════

# -- DSPy / RAG LM (used for answering questions) -----------------
lm = dspy.LM(
    'openai/%s' % os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv("OPEN_AI_KEY")
)
dspy.configure(lm=lm)

# -- Embedding function for ChromaDB ------------------------------
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
)

collection = chroma_client.get_or_create_collection(
    name="city_council",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1}
)

# -- Data ----------------------------------------------------------
df = pd.read_json(os.environ.get("FLY_DATA") + '/data.jsonl', lines=True)
default_start = df.date.min().strftime("%Y-%m-%d")
default_end   = df.date.max().strftime("%Y-%m-%d")

# -- RAG module ----------------------------------------------------
rag = RAG(collection)


# ═════════════════════════════════════════════════════════════════
# Flask app
# ═════════════════════════════════════════════════════════════════
app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Demo</title>
</head>
<body>
  <h1>Ask a question</h1>
  <p style="font-size:0.9em;"><a href="/admin">Admin</a></p>
  <form method="post">
    <textarea name="question" rows="4" cols="60" placeholder="Type your question here...">{{ question or '' }}</textarea><br>
    <label for="n_results">Max number of references:</label>
    <select name="n_results" id="n_results">
      {% for val in [5,10,15,20] %}
        <option value="{{ val }}"
          {% if val == n_results %}selected{% endif %}>
          {{ val }}
        </option>
      {% endfor %}
    </select>
    <br>
    <label for="start_date">Start Date:</label>
    <input type="date" id="start_date" name="start_date"
           value="{{ start_date or '' }}">&nbsp;&nbsp;
    <label for="end_date">End Date:</label>
    <input type="date" id="end_date" name="end_date"
           value="{{ end_date or '' }}"><br>
    <button type="submit">Ask</button>
  </form>
  {% if answer %}
    <h2>Answer:</h2>
    <div>{{ answer|safe }}</div>
  {% endif %}
</body>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = None
    n_results = 5
    start_date = default_start
    end_date   = default_end
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        start_date = request.form.get("start_date") or default_start
        end_date   = request.form.get("end_date")   or default_end
        try:
            n_results = int(request.form.get("n_results", 5))
        except ValueError:
            n_results = 5
        if question:
            result = rag(
                question=question,
                n_results=n_results,
                start_date=start_date,
                end_date=end_date,
            )
            answer = result.response + '<br><br>\n' + format_citations(result)
    return render_template_string(
        HTML_TEMPLATE,
        question=question,
        answer=answer,
        n_results=n_results,
        start_date=start_date,
        end_date=end_date,
    )


# ═════════════════════════════════════════════════════════════════
# Admin page — manage dictionaries and hardcoded corrections
# ═════════════════════════════════════════════════════════════════

ADMIN_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin — Dictionary Management</title>
</head>
<body>
  <h1>Admin — Dictionary Management</h1>
  <p><a href="/">← Back to search</a></p>

  {% if message %}
    <div style="padding:0.5em 1em; margin:1em 0; background:#d4edda; border:1px solid #c3e6cb;">
      {{ message }}
    </div>
  {% endif %}

  <hr>
  <h2>Add a Name</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_name">
    <label for="name_value">Name:</label>
    <input type="text" id="name_value" name="name_value" size="30" required>
    &nbsp;
    <label for="name_type">Type:</label>
    <select name="name_type" id="name_type">
      <option value="first">First Name</option>
      <option value="last">Last Name</option>
    </select>
    &nbsp;
    <button type="submit">Add Name</button>
  </form>
  <p><small>Currently: {{ n_first }} first names, {{ n_last }} last names</small></p>

  <hr>
  <h2>Add a Street</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_street">
    <label for="street_value">Street name (e.g. "Tchoupitoulas Street"):</label><br>
    <input type="text" id="street_value" name="street_value" size="50" required>
    &nbsp;
    <button type="submit">Add Street</button>
  </form>
  <p><small>Currently: {{ n_streets }} streets</small></p>

  <hr>
  <h2>Add a Hardcoded Correction</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_hardcoded">
    <label for="misspelling">Misspelling (e.g. "Geruso"):</label>
    <input type="text" id="misspelling" name="misspelling" size="25" required>
    &nbsp;
    <label for="correct_spelling">Correct spelling (e.g. "Giarrusso"):</label>
    <input type="text" id="correct_spelling" name="correct_spelling" size="25" required>
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

</body>
</html>
'''


# Load dictionaries for admin page display only
_app_dir = os.path.dirname(os.path.abspath(__file__))
_admin_dicts = load_dictionaries(
    english_path=os.path.join(_app_dir, "english_words.json"),
    names_path=os.path.join(_app_dir, "nola_names.json"),
    streets_path=os.path.join(_app_dir, "nola_streets.json"),
)


@app.route("/admin", methods=["GET", "POST"])
def admin():
    global _admin_dicts
    message = None
    app_dir = os.path.dirname(os.path.abspath(__file__))

    if request.method == "POST":
        action = request.form.get("action")

        # ── Add a name ──────────────────────────────────────────
        if action == "add_name":
            name_value = request.form.get("name_value", "").strip()
            name_type = request.form.get("name_type", "last")

            if name_value:
                names_path = os.path.join(app_dir, "nola_names.json")
                with open(names_path, "r", encoding="utf-8") as f:
                    names_data = json.load(f)

                key = "first_names" if name_type == "first" else "last_names"

                if name_value not in names_data.get(key, []):
                    names_data.setdefault(key, []).append(name_value)
                    with open(names_path, "w", encoding="utf-8") as f:
                        json.dump(names_data, f, indent=2, ensure_ascii=False)

                    _admin_dicts = load_dictionaries(
                        english_path=os.path.join(app_dir, "english_words.json"),
                        names_path=names_path,
                        streets_path=os.path.join(app_dir, "nola_streets.json"),
                    )
                    message = f"Added {name_type} name: {name_value}"
                else:
                    message = f"'{name_value}' already exists in {key}"

        # ── Add a street ────────────────────────────────────────
        elif action == "add_street":
            street_value = request.form.get("street_value", "").strip()

            if street_value:
                streets_path = os.path.join(app_dir, "nola_streets.json")
                with open(streets_path, "r", encoding="utf-8") as f:
                    streets_data = json.load(f)

                if street_value not in streets_data:
                    streets_data.append(street_value)
                    with open(streets_path, "w", encoding="utf-8") as f:
                        json.dump(streets_data, f, indent=2, ensure_ascii=False)

                    _admin_dicts = load_dictionaries(
                        english_path=os.path.join(app_dir, "english_words.json"),
                        names_path=os.path.join(app_dir, "nola_names.json"),
                        streets_path=streets_path,
                    )
                    message = f"Added street: {street_value}"
                else:
                    message = f"'{street_value}' already exists in streets"

        # ── Add a hardcoded correction ──────────────────────────
        elif action == "add_hardcoded":
            misspelling = request.form.get("misspelling", "").strip()
            correct_spelling = request.form.get("correct_spelling", "").strip()

            if misspelling and correct_spelling:
                hc_path = os.path.join(app_dir, "hardcoded_corrections.json")

                try:
                    with open(hc_path, "r", encoding="utf-8") as f:
                        hc_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    hc_data = {}

                hc_data[misspelling.lower()] = correct_spelling

                with open(hc_path, "w", encoding="utf-8") as f:
                    json.dump(hc_data, f, indent=2, ensure_ascii=False)

                load_hardcoded_corrections(hc_path)
                message = f"Added hardcoded correction: '{misspelling}' → '{correct_spelling}'"

    # ── Gather current counts for display ───────────────────────
    from correction import HARDCODED_CORRECTIONS

    names_path = os.path.join(app_dir, "nola_names.json")
    try:
        with open(names_path, "r", encoding="utf-8") as f:
            nd = json.load(f)
        n_first = len(nd.get("first_names", []))
        n_last = len(nd.get("last_names", []))
    except Exception:
        n_first = n_last = 0

    hardcoded_list = sorted(HARDCODED_CORRECTIONS.items())

    return render_template_string(
        ADMIN_TEMPLATE,
        message=message,
        n_first=n_first,
        n_last=n_last,
        n_streets=len(_admin_dicts.get("streets", [])),
        n_hardcoded=len(HARDCODED_CORRECTIONS),
        hardcoded_list=hardcoded_list,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)