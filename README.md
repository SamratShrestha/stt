# STT (Speech-to-Text) API

A FastAPI-based speech-to-text service with Docker support.

## Quick Start

### Development

1. Clone the repository
2. Copy `.env` and update configuration as needed
3. Run the application:

```bash
docker-compose up --build
```

OR

```bash
uv sync
uv run fastapi dev app/main.py;
```

The API will be available at `http://localhost:8000`

### Production

Set `ENVIRONMENT=production` in your environment variables and deploy.

## Docker Commands

```bash
# Initial build and run
docker-compose up --build

# Run without rebuilding (for code changes)
docker-compose up

# Stop containers
docker-compose down

# View logs
docker-compose logs -f