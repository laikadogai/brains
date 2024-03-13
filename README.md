# Brains

Control center of Robokeeper. Constantly listens to wake up phrase and executes tasks.

## How to Run

Install dependencies:

```bash
poetry install
```

Make sure you have `OPENAI_API_KEY` and `ELEVEN_API_KEY` environment variables, and run service locally via:

```bash
poetry run python app.py
```

It is possible to control script parameters via environment variables listed in [Args class](./brains/__init__.py), for example:

```bash
openai_temperature=0.7 poetry run python app.py
```

### Docker Version (not working yet)

Alternatively you can build and run service via [docker-compose.yml](./docker-compose.yml) or with plain `docker` similar to this:

```bash
sudo docker buildx build --tag brains .
sudo docker run --tty --name brains --rm \
    --env OPENAI_API_KEY=${OPENAI_API_KEY} \
    --env ELEVEN_API_KEY=${ELEVEN_API_KEY} \
    --volume $PWD/data:/app/data \
    --volume $PWD/models:/app/models \
    brains
```

## ToDo

- Make service work in Docker by figuring out hardware forwarding
- Listen until no sound and get rid of hardcoded window
- Write messaging state to local storage to preserve between restarts
- When within ongoing conversation thread, don't wait for "activation phrase"
- Remove "Hey, robot" from messages history to avoid clutter (maybe)
- Add some kind of cron jobs, e.g. rescan the environment

## Miscellaneous

### Local Modules vs Online APIs

It is possible to replase ASR with Whisper and TTS with the best open model from [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena).
