FROM python:3.11

# Install system dependencies required by PyAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio

ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN poetry install

# Preload and cache openwakeword preprocessing models
RUN poetry run python -c "import openwakeword; openwakeword.utils.download_models()"

COPY . /app
CMD [ "poetry", "run", "python", "app.py" ]
