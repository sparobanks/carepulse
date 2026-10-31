.PHONY: install test results quick roads serve llm-eval

install:
	pip install -r requirements.txt

test:
	pytest -q

results:
	python scripts/run_all.py

quick:
	python scripts/run_all.py --quick

roads:  ## download real Birmingham roads (needs internet, once)
	mkdir -p data && python -c "from carepulse.geo.travel import RoadNetworkTravel as R; R.download('Birmingham, UK', 'data/birmingham_drive.graphml')"

serve:
	CAREPULSE_DEMO=1 uvicorn carepulse.api.app:app --reload

llm-eval:
	python scripts/eval_llm.py --model llama3.1:8b
