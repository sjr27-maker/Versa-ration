FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY . .
RUN uv sync --frozen
ENV PORT=8080
EXPOSE 8080
# `probe serve` binds 0.0.0.0 and reads $PORT by default (see cli.py).
CMD ["uv", "run", "probe", "serve"]
