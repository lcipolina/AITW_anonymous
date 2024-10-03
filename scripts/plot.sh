# Examples how to produce various plots from raw data, as used in the paper   




SCRIPT_PATH="scripts/plot_figures.py"
PROMPTS_JSON="prompts/prompts.json"
JSONS_DIR="collected_responses"



# Plot Fig. 1 (main boxplot)
# Box plot corresponding to Fig. 1 showing correct response rates averaged over AIW variations 1-4
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



SCRIPT_PATH="scripts/plot_figures.py"
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


# Plot Fig. 1 (inlay barplot showing performance fluctuations across AIW variations 1-4)
# --prompts_ids should contain prompt IDs as listed in prompts.json
# correct response rate is plotted as a vertical bar for each of AIW variation 1-4
# Inlay in Figure 1 shows fluctuations for STANDARD prompt type for GPT-4
# STANDARD: 55 56 63 69
#  --model_list_path shows to the file containing models to plot
# raw data is assumed to be in --jsons_dir PATH
# prompt list is assumed to be in --prompts_json PATH

OUTPUT_PATTERN="fig1_barplot_AIW_fluctuations_Var_1-4_STANDARD_{}.pdf"
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
# Plot Fig. 2, barplot showing performance fluctuations across AIW variations 1-4 for THINKING prompt type
# --prompts_ids should contain prompt IDs as listed in prompts.json
# correct response rate is plotted as a vertical bar for each of AIW variation 1-4
# Figure 2 shows fluctuations for THINKING prompt type for various selected models
# THINKING: 57 58 64 70
#  --model_list_path shows to the file containing models to plot
# raw data is assumed to be in --jsons_dir PATH
# prompt list is assumed to be in --prompts_json PATH


OUTPUT_PATTERN="fig2_AIW_fluctuations_Var_1-4_THINKING_{}.pdf"
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

# Fig. 3 - correct response rate barplot 
# AIW LIGHT Control Arithmetic Total Siblings THINKING v2 
# Prompt IDS: 277 278 279 280
OUTPUT_PATTERN="fig3_AIW_Light_Control_Arithmetic_Total_Siblings_THINKING_v2_{}.pdf"
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

# Fig. 4 - correct response rate barplot 
# AIW LIGHT Control Family Alice's Sister's Brothers THINKING v2 
# Prompt IDS: 271 272 273 274
OUTPUT_PATTERN="fig4_AIW_Light_Control_Family_THINKING_v2_{}.pdf"
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

# Fig. 5 - correct response rate barplot 
# AIW LIGHT Control Arithmetic Total Girls THINKING v2 
# Prompt IDS: 343 344 345 346
OUTPUT_PATTERN="fig5_AIW_Light_Control_Arithmetic_Total_Girls_THINKING_v2_{}.pdf"
MODEL_LIST="models/models_set_AIW_Light_Control.json"
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


# Fig. 6a - barplot, correct response rates across AIW variations 1-4, fluctuations
# AIW Alice Female Power Boost ("Alice is female ..."), THINKING v2
# prompt IDs: 193 197 189 190
OUTPUT_PATTERN="fig6a_AIW_Alice_Female_Power_Boost_THINKING_v2_{}.pdf"
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

# Fig. 6b - barplot, correct response rates across AIW variations 1-4, fluctuations
# AIW THINKING v2
# prompt IDs: 205 206 187 188 

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

# Fig. 7 - stanadtized benchmarks vs AIW (55 56 63 69 57 58 64 70)
# Colors can differ from the original plot
OUTPUT_PATTERN="fig7_{}.pdf"
MODEL_LIST="models/models_plot_set_v1.json"

CMD="python3 $SCRIPT_PATH \
    --output $OUTPUT_PATTERN \
    --jsons_dir $JSONS_DIR \
    --prompt_ids 55 56 63 69 57 58 64 70  \
    --title "" \
    --prompts_json $PROMPTS_JSON \
    --model_list_path $MODEL_LIST \
    --fig_width 15 \
    --fig_height 12 \
    --axis_fontsize 20 \
    --title_fontsize 12 \
    --plot_scatter"

$CMD && echo "Plotting done" || echo "Plotting failed"
