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

chroma_client = PersistentClient(
    path='/models/chroma_db',            # where on disk to store
    settings=Settings(anonymized_telemetry=False)
)

class RAGQuestion(dspy.Signature):
    """
    Answer this question about New Orleans City Council meetings based only on the provided context.
    Return a citation-aware response.
    """
    question: str = dspy.InputField()
    # Cite the sources you use with citation labels like [CITATION 1].
    context: str = dspy.InputField() # desc='Passages with citation labels like [CITATION 1], [CITATION 2]')
    # context: list[str] = dspy.InputField()
    response: str = dspy.OutputField(desc='Answer with or without inline citations like [CITATION 1]')
    citations: list[str] = dspy.OutputField(desc="List of citation sources used, e.g., [[CITATION 1], [CITATION 2]]")
    
    
class RAG(dspy.Module):
    
    def __init__(self, collection):
        self.collection = collection
        self.respond = dspy.ChainOfThought(RAGQuestion)

    def forward(self, question, n_results=5):
        result = self.collection.query(query_texts=[question], n_results=n_results)
        context = result['documents'][0]
        context = '\n\n'.join('### [CITATION %i]\n%s' % (i,s) for i,s in enumerate(context))
        response = self.respond(context=context, question=question) 
        response['context'] = context
        response['documents'] = result['documents'][0]
        response['meta'] = result['metadatas'][0]
        response['distances'] = result['distances'][0]
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

def citation2html(citation_no, row, start_time, txt, summary):
    video_num = 'video%d' % citation_no
    return """
    <details>
      <summary><strong>Reference %d</strong></summary>
      <div style="padding:0.5em 1em;">
      <p>%s (%s)</p>
      <p>%s</p>
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
      <details>
        <summary>Transcript</summary>
        <div style="padding:0.5em 1em;">
          <p>%s</p>
        </div>
      </details>    
      </div>
    </details>
    """ % (
        citation_no,
        row.title, row.date,
        markdown.markdown(summary, extensions=["fenced_code", "tables"]),
        video_num, row.box_link, video_num, video_num, 
        start_time,
        txt
        )

def format_citations(result):
    citations = []
    print(result['meta'])
    for c in result.citations:
        num = int(re.findall('([\d+])', c)[0])
        meta = result['meta'][num]
        # jfile = PATH + re.sub('.summary', '.json', meta['file'].split('/')[-1])
        mfile = re.sub('.summary', '.mp4', meta['file'].split('/')[-1])
        # js = [json.loads(j) for j in open(jfile)]
        txt = 'tbd' # get_text_by_ids(js, meta['start_id'], meta['end_id'])
        row = df[df.video.str.contains(mfile)].iloc[0]
        citations.append(
            citation2html(num, row, meta['start_time'], txt, result['documents'][num]))
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
df = pd.read_json('/models/data.jsonl', lines=True)
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
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
             result = rag(question=question, n_results=5)
             answer = result.response + '<br><br>\n' + format_citations(result)
    return render_template_string(HTML_TEMPLATE, question=question, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
