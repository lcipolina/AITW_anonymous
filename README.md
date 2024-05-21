# Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models

## Repo structure

```
.
├── collected_responses 
├── models_data # data on benchmarks and additional info like HF Hub models aliases
├── prompts
│   ├── prompts.json
├── scripts
│   ├── plot.py
│   ├── plot.sh # bash script for paper plot generation
```

## Usage

Install requirements:
`pip install requirements.txt`

Run script to generate plots from the paper (by deafault prompts will be saved in the working directory):
`bash scripts/plot.sh`

