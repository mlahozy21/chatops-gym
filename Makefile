.PHONY: up down generate validate test eval report

up:            ## start the environment
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

test:          ## unit tests (no server needed)
	python -m pytest tests/ -q

generate:      ## build task instances from templates
	python -m scripts.generate_tasks --per-template 5

validate:      ## the harness: oracle=1, mutants=0, null=0, random=0
	python -m validation.validate_curriculum

eval:          ## run a real model (default: local Qwen via Ollama)
	python -m scripts.run_eval --backend ollama:qwen2.5:7b

report:        ## difficulty calibration for the backend you evaluated
	python -m calibration.report results/ollama_qwen2.5_7b.jsonl
