# nocouncil

Testing sandbox for a RAG system for searching City Council video transcripts.

Reads Chroma vector database from CHROMA_URL and serves with Open AI.

Running at https://nocouncil.fly.dev/

See data processing at https://github.com/tapilab/nocouncil-etl

### local setup

To run on your local machine:

**Prerequisites**
- Python version 3.9
- pip
- virtualenv
  
**Installation**

```sh
mkdir nocouncil-root
cd nocouncil-root
# make Python virtual environment
virtualenv --python=python3.9 venv
source venv/bin/activate
# clone repo
git clone https://github.com/tapilab/nocouncil.git
cd nocouncil
# install Python requirements. This can take a while.
pip install -r requirements.txt 
```

Create a file called `.env` in the `nocouncil` folder. Paste in the `.env` file available [here](https://drive.google.com/drive/u/0/folders/12LjWYraAbP5pnLjob_c6FufouQPIdWUG)
This sets the URL to read the Chroma database from Box, the OpenAI key, as well as port the web server will use.

Now, you can run 

`./entrypoint`

You should see something like:

```
[2025-09-29 21:03:41 -0500] [10087] [INFO] Starting gunicorn 23.0.0
[2025-09-29 21:03:41 -0500] [10087] [INFO] Listening at: http://0.0.0.0:5001 (10087)
[2025-09-29 21:03:41 -0500] [10087] [INFO] Using worker: sync
[2025-09-29 21:03:41 -0500] [10094] [INFO] Booting worker with pid: 10094
```
You should then see the website running locally at http://0.0.0.0:5001

**Errors:**

- `[Errno 48] Address already in use`: Try a different port (in .env); this one's already being used.

### fly.io setup

- 1. Install the Fly CLI 
curl -L https://fly.io/install.sh | sh

(you may need to restart your shell or add $HOME/.fly/bin to your PATH)

- 2. Authenticate: `flyctl auth login`

- 3. If using existing app, ask owner to add you as collaborator. Otherwise, you'll need to create a new Fly app:
  - `flyctl apps create my-app-name`
  - generate fly.toml `flyctl init --name my-app-name --no-deploy`
  - provision data `flyctl volumes create data --size 2`

- 4. Set secrets: `flyctl secrets import < .env`

- 5. Deploy: `flyctl deploy`
