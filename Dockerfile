FROM python:3.11-slim

# Run the application as a non-root user instead of the image's default `root`
# user, limiting the blast radius of a container compromise.
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home --home-dir /home/appuser appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application package. Sensitive files (.env, secrets, local
# caches) are excluded by .dockerignore, but copying the app directory
# explicitly avoids ever baking unrelated repo contents into the image.
COPY app ./app

# The application code is read-only at runtime, but make sure the non-root
# user can read it.
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
