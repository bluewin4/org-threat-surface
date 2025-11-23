quick-test:
	uv run main.py --mode sweep --n-orgs 5 --repeats 5 --steps 100
medium-test:
	uv run main.py --mode sweep --n-orgs 100 --repeats 50 --steps 100 --max-nodes 50
long-test:
	uv run main.py --mode sweep --n-orgs 500 --repeats 50 --steps 100 --max-nodes 50
last-test:
	uv run main.py --mode sweep --n-orgs 50 --repeats 10 --steps 100 --max-nodes 50
dashboard:
	uv run dashboard_server.py