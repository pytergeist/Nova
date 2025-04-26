# ────────────────
# 1. Builder dependancies for nove and compile fusion
# ────────────────
FROM python:3.13-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      python3-dev \
      pybind11-dev \
      libopenblas-dev \
      liblapack-dev \
      libeigen3-dev \
      libgtest-dev \
 && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 \
 && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

COPY . .
WORKDIR /app/fusion
RUN bash compile.sh


# ────────────────
FROM python:3.13-slim


RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libstdc++6 \
      libopenblas0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin         /usr/local/bin

WORKDIR /app
COPY --from=builder /app/fusion/build /app/fusion/build
COPY --from=builder /app               /app

ENV PYTHONPATH=/app

ENTRYPOINT ["pytest"]
CMD ["-v"]
