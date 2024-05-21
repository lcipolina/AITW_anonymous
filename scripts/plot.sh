OUTPUT_PATTERN="{}_prompt.pdf"

JSONS_DIR="collected_responses"
SCRIPT_PATH="scripts/plot.py"
PROMPT_ID="55,56,69"
PROMPTS_JSON="prompts/prompts.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_id $PROMPT_ID \
    --prompts_json $PROMPTS_JSON"

echo $CMD
$CMD