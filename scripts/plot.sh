# plot version similar to Fig. 1
# --prompts_ids should contain prompt IDs as listed in prompts.json
# as we average across AIW variation 1-4, these are groups of 4 IDs, 
# where each group stands for using particular prompt type,
# eg plotting average correct response rates for STANDARD and THINKING, 
# as in Figure 1:
# STANDARD: 55 56 63 69 
# THINKING: 57 58 64 70
#  --model_list_path shows to the file containing models to plot
# raw data is assumed to be in --jsons_dir PATH
# prompt list is assumed to be in --prompts_json PATH



SCRIPT_PATH="scripts/plot_resp.py"
PROMPTS_JSON="prompts/prompts.json"
JSONS_DIR="collected_responses"



# Plot version similar to Fig. 1 (boxplot)

OUTPUT_PATTERN="fig1_{}.pdf"
MODEL_LIST="models/models_plot_set_FULL.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 55 56 63 69 57 58 64 70 \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST\
    --title "" \
    --fig_width 8 \
    --fig_height 4 \
    --axis_fontsize 8 \
    --title_fontsize 11 \
    --plot_boxplot"

$CMD && echo "Plotting done" || echo "Plotting failed"


# Plot version similar to Fig. 1 (barplot)

OUTPUT_PATTERN="fig1_barplot_{}.pdf"
MODEL_LIST="models/gpt-4.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 55 56 63 69 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 4 \
    --fig_height 6 \
    --axis_fontsize 16 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"

# Fig. 2 - fluctuation of correct response rate for THINKING v1 (57 58 64 70)
OUTPUT_PATTERN="fig2_{}.pdf"
MODEL_LIST="models/models_set_fluctuations_THINKING.json"
CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 57 58 64 70 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed" 

# Fig. 3 - fluctuation of correct response rate for Control THINKING v2 (277 278 279 280)
OUTPUT_PATTERN="fig3_{}.pdf"
MODEL_LIST="models/models_set_AIW_Light_Control.json"
CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 277 278 279 280 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"

# Fig. 4 - fluctuation of correct response rate for Control THINKING v2 (271 272 273 274)
OUTPUT_PATTERN="fig4_{}.pdf"
MODEL_LIST="models/models_set_AIW_Light_Control.json"
CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 271 272 273 274 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"

# Fig. 5 - fluctuation of correct response rate for Control THINKING v2 Arithmetic (343 344 345 346)
OUTPUT_PATTERN="fig5_{}.pdf"
MODEL_LIST="models/models_set_AIW_Light_Control_arithmetic.json"
CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 343 344 345 346 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"


# Fig. 6a - fluctuation of correct response rate for Alice Female boost THINKING v2 (193 197 189 190)
OUTPUT_PATTERN="fig6a_{}.pdf"
MODEL_LIST="models/models_set_Female_Boost_Thinking_v2.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 193 197 189 190 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"

# # Fig. 6b - fluctuation of correct response AIW Original THINKING v2 (205 206 187 188)
OUTPUT_PATTERN="fig6b_{}.pdf"
MODEL_LIST="models/models_set_Female_Boost_Thinking_v2.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 205 206 187 188 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 20 \
    --fig_height 8 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_multiple \
    --show_variation"

$CMD && echo "Plotting done" || echo "Plotting failed"

# Fig. 7 - stanadtized benchmarks vs AIW (55 56 63 69 57 58 64 70 53 54 65 71)
# Colors can differ from the original plot
OUTPUT_PATTERN="fig7_{}.pdf"
MODEL_LIST="models/models_plot_set_v1.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 55 56 63 69 57 58 64 70 53 54 65 71 \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 15 \
    --fig_height 12 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_scatter"

$CMD && echo "Plotting done" || echo "Plotting failed"
