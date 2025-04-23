# nocouncil

Testing sandbox for a RAG system for searching City Council video transcripts.

Pulls

### configuring fly.io

- 1. Install the Fly CLI 
curl -L https://fly.io/install.sh | sh

(you may need to restart your shell or add $HOME/.fly/bin to your PATH)

- 2. Authenticate: `flyctl auth login`

- 3. If using existing app, ask owner to add you as collaborator. Otherwise, you'll need to create a new Fly app:
  - `flyctl apps create my-app-name`
  - generate fly.toml `flyctl init --name my-app-name --no-deploy`
  - provision data `flyctl volumes create data --size 2`

- 4. Set secrets: `flyctl secrets import < .env`