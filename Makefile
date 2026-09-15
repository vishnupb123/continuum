.PHONY: up down logs ps reset api-shell test

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

reset:
	docker compose down -v

db-migrate:
	docker compose run --rm api alembic upgrade head

test:
	docker compose run --rm api pytest -q
