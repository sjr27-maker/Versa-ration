FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY . .
RUN uv sync --frozen
ENV PORT=8080
EXPOSE 8080
CMD ["uv", "run", "probe", "web"]