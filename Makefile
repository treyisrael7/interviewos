.PHONY: up down up-verbose up-minimal logs db-migrate build clean retrieval-eval retrieval-eval-compare

up:
	docker compose up -d --build

# Run with visible progress (no -d) - use this when up seems stuck
up-verbose:
	docker compose build --progress=plain && docker compose up

# Postgres + API only (skips heavy web build) - use to verify Docker works
up-minimal:
	docker compose up -d --build postgres api

# Build one service at a time (helps if builds hang from resource limits)
up-serial:
	COMPOSE_PARALLEL_LIMIT=1 docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

db-migrate:
	docker compose exec api alembic upgrade head

build:
	docker compose build

test:
	python -c "import subprocess,time;r=subprocess.run(['docker','compose','ps','postgres','-q'],capture_output=True,text=True);(r.returncode!=0 or not r.stdout.strip()) and (subprocess.run(['docker','compose','up','-d','postgres']),time.sleep(5))"
	cd apps/api && alembic upgrade head && pytest -v

test-docker:
	docker compose build api && docker compose run --rm api sh -c "alembic upgrade head && python -m pytest -v"

retrieval-eval:
	cd apps/api && python -m evals.retrieval.entrypoint $(ARGS)

retrieval-eval-compare:
	cd apps/api && python -m evals.retrieval.entrypoint --compare $(ARGS)

clean:
	docker compose down -v
