"""
Simple Flask app to serve a RAG model over video transcripts.
- Reads chroma vector database from CHROMA_DB_DIR
- Requires OPENAI_API_KEY
- model set by OPENAI_MODEL (default: gpt-4o-mini)
- port set by PORT (8080 default)

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

# load_dotenv()

chroma_client = PersistentClient(
    path=os.getenv('CHROMA_DB_DIR', '/models/chroma_db'),# where on disk to store
    settings=Settings(anonymized_telemetry=False)
)

class RAGQuestion(dspy.Signature):
    """
    Answer this question about New Orleans City Council meetings based only on the provided context.
    Cite as many relevant sources as you can with citation labels like [CITATION 1].
    """
    question: str = dspy.InputField()
    # Cite the sources you use with citation labels like [CITATION 1].
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
        # convert to the same numeric type you stored
        start_ts = int(start_dt.timestamp())
        end_ts   = int(end_dt.timestamp())
        where = {
          "$and": [
            {"date": {"$gte": start_ts}},
            {"date": {"$lte": end_ts}}
          ]
        }
        result = self.collection.query(query_texts=[question], 
            n_results=n_results, where=where)
        context = result['documents'][0]
        context = '\n\n'.join('### [CITATION %i]\n%s' % (i,s) for i,s in enumerate(context))
        response = self.respond(context=context, question=question) 
        response['context'] = context
        response['ids'] = result['ids'][0]
        response['documents'] = result['documents'][0]
        response['meta'] = result['metadatas'][0]
        # response['distances'] = result['distances'][0]
        return response

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
            // Jump to time in seconds
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
        num = int(re.findall('([\d+])', c)[0])
        if num in cites_seen: # skip
            continue
        cites_seen.add(num)
        meta = result['meta'][num]
        # jfile = PATH + re.sub('.summary', '.json', meta['file'].split('/')[-1])
        mfile = re.sub('.summary', '.mp4', meta['file'].split('/')[-1])
        # js = [json.loads(j) for j in open(jfile)]
        quotes = '<ul>'
        for h in meta['quotes'].split('|||')[:3]:
            quotes += '\n<li>"%s"</li>' % h
        quotes += '\n</ul>\n'
        names = ', '.join(sorted(meta['names'].split('|||')))
        row = df[df.video.str.contains(mfile)].iloc[0]
        citations.append(
            citation2html(i+1, num, row, meta['start_time'], quotes, names, result['documents'][num]))
    return '\n<br>\n'.join(citations)
        

lm = dspy.LM('openai/%s' % os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
             api_key=os.getenv("OPEN_AI_KEY"))
dspy.configure(lm=lm)
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # small, fast, 384‑dim
    device="cpu",                   # or "cuda"
    normalize_embeddings=True,
)

collection = chroma_client.get_or_create_collection(
        name="city_council",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine",
                  "hnsw:num_threads": 1}
        )
df = pd.read_json(os.environ.get("FLY_DATA") + '/data.jsonl', lines=True)
default_start = df.date.min().strftime("%Y-%m-%d")
default_end   = df.date.max().strftime("%Y-%m-%d")
rag = RAG(collection)

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
             result = rag(question=question, n_results=n_results,
                start_date=start_date, end_date=end_date)
             # answer = re.sub(r'[\[\(]CITATION \d+[\]\)]', ' ', result.response) + '<br><br>\n' + format_citations(result)
             answer = result.response + '<br><br>\n' + format_citations(result)
    return render_template_string(HTML_TEMPLATE, question=question, 
        answer=answer, n_results=n_results,start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
