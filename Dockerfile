FROM python:3.12-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
# preload_docs: auto-loaded on startup when collection is empty
# demo_docs: selectable from dropdown to add during demo
COPY src/preload_docs ./preload_docs
COPY src/demo_docs ./demo_docs

EXPOSE 8080
ENV PORT=8080

CMD ["python", "app.py"]
