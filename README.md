# Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models

## Repo structure

```
.
├── collected_responses
│   ├── raw_data_inspection # varios raw data collections for viewing model responses
│   ├── AIW_responses.json.gz # all collected raw data archived, for plotting. Can be unpacked for viewing 
├── models_data # data on benchmarks and additional info like HF Hub models aliases
├── prompts
│   ├── prompts.json # full prompts with IDs used for the AIW experiments
├── models # folder of model lists
├── scripts
│   ├── plot.py
│   ├── plot.sh # bash script showing examples for plot generation
```

## Usage

Install requirements:
`pip install -r requirements.txt`

Run script to generate main plots from the paper (by default plots will be saved in the working directory):
`bash scripts/plot.sh`
