
"""
Simple Flask app to serve a RAG model over video transcripts AND news articles.
- Searches both city_council and articles collections
- Requires OPENAI_API_KEY
- model set by OPENAI_MODEL (default: gpt-4o-mini)
"""
import os
from datetime import datetime
import dspy
from flask import Flask, request, jsonify, render_template_string
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from dotenv import load_dotenv
import json
import markdown
import openai
import pandas as pd
import re

chroma_client = PersistentClient(
    path=os.getenv('CHROMA_DB_DIR', '/models/chroma_db'),
    settings=Settings(anonymized_telemetry=False)
)

class RAGQuestion(dspy.Signature):
    """
    Answer this question about New Orleans City Council meetings and civic news based only on the provided context.
    Cite as many relevant sources as you can with citation labels like [CITATION 1].
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='Passages with citation labels like [CITATION 1], [CITATION 2].')
    response: str = dspy.OutputField(desc='Answer without inline citations.')
    citations: list[str] = dspy.OutputField(desc="List of citation sources used, e.g., [[CITATION 1], [CITATION 2]]. Cite as many as are relevant.")

def filename2date(filename):
    mp4 = re.sub('.summary', '.mp4', filename.split('/')[-1])
    return df[df.video.str.contains(mp4)].iloc[0].date

class DualRAG(dspy.Module):
    """RAG that searches both city council transcripts and news articles"""
    
    def __init__(self, council_collection, articles_collection):
        self.council_collection = council_collection
        self.articles_collection = articles_collection
        self.respond = dspy.ChainOfThought(RAGQuestion)

    def forward(self, question, start_date, end_date, n_results=5, source_type="both"):
        """
        source_type: "council", "articles", or "both"
        """
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
        all_sources = []  # Track which collection each result came from
        
        # Search city council if requested and available
        if source_type in ["council", "both"] and self.council_collection is not None:
            try:
                council_results = self.council_collection.query(
                    query_texts=[question], 
                    n_results=n_results, 
                    where=where
                )
                all_documents.extend(council_results['documents'][0])
                all_ids.extend(council_results['ids'][0])
                all_meta.extend(council_results['metadatas'][0])
                all_sources.extend(['council'] * len(council_results['documents'][0]))
            except Exception as e:
                print(f"Error querying council: {e}")
        
        # Search articles if requested and available
        if source_type in ["articles", "both"] and self.articles_collection is not None:
            try:
                article_results = self.articles_collection.query(
                    query_texts=[question], 
                    n_results=n_results
                )
                all_documents.extend(article_results['documents'][0])
                all_ids.extend(article_results['ids'][0])
                all_meta.extend(article_results['metadatas'][0])
                all_sources.extend(['article'] * len(article_results['documents'][0]))
            except Exception as e:
                print(f"Error querying articles: {e}")
        
        # If no results, return empty response
        if not all_documents:
            return {
                'response': "I couldn't find any relevant information to answer your question.",
                'context': '',
                'ids': [],
                'documents': [],
                'meta': [],
                'sources': [],
                'citations': []
            }
        
        # Build context with citations
        context = '\n\n'.join(
            '### [CITATION %i] (Source: %s)\n%s' % (i, src.upper(), doc) 
            for i, (doc, src) in enumerate(zip(all_documents, all_sources))
        )
        
        response = self.respond(context=context, question=question) 
        response['context'] = context
        response['ids'] = all_ids
        response['documents'] = all_documents
        response['meta'] = all_meta
        response['sources'] = all_sources

        return response

def citation2html_council(i, citation_no, row, start_time, quotes, names, summary):
    """Format citation for city council video"""
    video_num = 'video%d' % citation_no
    return """
    <details>
      <summary><strong>Reference %d [CITY COUNCIL VIDEO]</strong></summary>
      <div style="padding:0.5em 1em;">
      <p>%s (%s)</p>
      <p>%s</p>
      <p><i>Quotes</i><br>
          %s
      </p>
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
    """ % (
        i, row.title, str(row.date)[:10],
        markdown.markdown(summary, extensions=["fenced_code", "tables"]),
        quotes, names, video_num, row.box_link, video_num, video_num, start_time
    )

def citation2html_article(i, meta, summary):
    """Format citation for news article"""
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
    """ % (
        i,
        meta.get('title', 'Untitled'),
        meta.get('source', 'Unknown'),
        meta.get('published', '')[:10],
        markdown.markdown(summary[:500] + '...' if len(summary) > 500 else summary),
        meta.get('url', '#')
    )

def format_citations(result):
    """Format citations from both council and article sources"""
    citations = []
    cites_seen = set()
    
    for i, c in enumerate(result.citations):
        num = int(re.findall('([\d+])', c)[0])
        if num in cites_seen:
            continue
        cites_seen.add(num)
        
        if num >= len(result['sources']):
            continue
            
        source_type = result['sources'][num]
        meta = result['meta'][num]
        doc = result['documents'][num]
        
        if source_type == 'council':
            # City council citation with video
            mfile = re.sub('.summary', '.mp4', meta['file'].split('/')[-1])
            quotes = '<ul>'
            for h in meta['quotes'].split('|||')[:3]:
                quotes += '\n<li>"%s"</li>' % h
            quotes += '\n</ul>\n'
            names = ', '.join(sorted(meta['names'].split('|||')))
            row = df[df.video.str.contains(mfile)].iloc[0]
            citations.append(
                citation2html_council(i+1, num, row, meta['start_time'], quotes, names, doc)
            )
        elif source_type == 'article':
            # News article citation
            citations.append(
                citation2html_article(i+1, meta, doc)
            )
    
    return '\n<br>\n'.join(citations)

# Initialize
lm = dspy.LM('openai/%s' % os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
             api_key=os.getenv("OPEN_AI_KEY"))
dspy.configure(lm=lm)

embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
)

# Load both collections (gracefully handle missing data)
try:
    council_collection = chroma_client.get_collection(
        name="city_council",
        embedding_function=embed_fn
    )
    print("Loaded city_council collection")
except Exception as e:
    print(f"City council collection not found: {e}")
    council_collection = None

try:
    articles_collection = chroma_client.get_collection(
        name="articles",
        embedding_function=embed_fn
    )
    print(f"Loaded articles collection ({articles_collection.count()} articles)")
except Exception as e:
    print(f"Warning: Articles collection not found: {e}")
    articles_collection = None

# Load dataframe for council metadata (optional)
try:
    df = pd.read_json(os.environ.get("FLY_DATA", './') + '/data.jsonl', lines=True)
    default_start = df.date.min().strftime("%Y-%m-%d")
    default_end   = df.date.max().strftime("%Y-%m-%d")
except Exception as e:
    print(f"Warning: Council metadata not found: {e}")
    df = None
    default_start = "2020-01-01"
    default_end = datetime.now().strftime("%Y-%m-%d")

rag = DualRAG(council_collection, articles_collection)

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NOLA Civic RAG</title>
</head>
<body>
  <h1>Ask a question about New Orleans City Council or Civic News</h1>
  <form method="post">
    <textarea name="question" rows="4" cols="60" placeholder="Type your question here...">{{ question or '' }}</textarea><br>
    
    <label for="source_type">Search in:</label>
    <select name="source_type" id="source_type">
      <option value="both" {% if source_type == 'both' %}selected{% endif %}>Both Council & News</option>
      <option value="council" {% if source_type == 'council' %}selected{% endif %}>City Council Only</option>
      <option value="articles" {% if source_type == 'articles' %}selected{% endif %}>News Articles Only</option>
    </select>
    <br>
    
    <label for="n_results">Max number of references:</label>
    <select name="n_results" id="n_results">
      {% for val in [5,10,15,20] %}
        <option value="{{ val }}" {% if val == n_results %}selected{% endif %}>{{ val }}</option>
      {% endfor %}
    </select>
    <br>
    
    <label for="start_date">Start Date:</label>
    <input type="date" id="start_date" name="start_date" value="{{ start_date or '' }}">&nbsp;&nbsp;
    <label for="end_date">End Date:</label>
    <input type="date" id="end_date" name="end_date" value="{{ end_date or '' }}"><br>
    
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
    end_date = default_end
    source_type = "both"
    
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
            result = rag(
                question=question, 
                n_results=n_results,
                start_date=start_date, 
                end_date=end_date,
                source_type=source_type
            )
            answer = result.response + '<br><br>\n' + format_citations(result)
    
    return render_template_string(
        HTML_TEMPLATE, 
        question=question, 
        answer=answer, 
        n_results=n_results,
        start_date=start_date, 
        end_date=end_date,
        source_type=source_type
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)