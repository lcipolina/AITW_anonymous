import pandas as pd
import numpy as np
from scipy import stats
import glob
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
import re
from adjustText import adjust_text
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

MODEL_MAPPINGS = {
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-2024-08-06": "GPT-4o-v2",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4-0613": "GPT-4",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "gpt-4-0125-preview": "GPT-4 Preview",
    "gpt-3.5": "GPT-3.5",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "claude-3-opus-20240229": "Claude-3 O",
    "claude-3-sonnet-20240229": "Claude-3 S",
    "claude-3-haiku-20240307": "Claude-3 H",
    "claude-3-5-sonnet-20240620": "Claude-3.5 S",
    "llama-2-70b-chat": "Llama-2 70b",
    "llama-2-13b-chat": "Llama-2 13b",
    "llama-2-7b-chat": "Llama-2 7b",
    "llama-3-8b-chat": "Llama-3 8b",
    "llama-3-70b-chat": "Llama-3 70b",
    "meta-llama-3.1-405b-instruct": "Llama-3.1 405b",
    "reflection-llama-3.1-70b": "Reflection-70b",
    "llama-3-70b-instruct": "Llama-3 70b",
    "llama-3-8b-instruct": "Llama-3 8b",
    "codellama-70b-instruct": "Codellama 70b",
    "mistral-large-2402": "Mistral Large",
    "mistral-large-latest": "Mistral Large",
    "mistral-medium-2312": "Mistral Medium",
    "open-mixtral-8x22b-instruct-v0.1": "Mixtral 8x22b",
    "open-mixtral-8x7b-instruct": "Mixtral 8x7b",
    "open-mistral-7b-instruct": "Mistral 7b",
    "open-mistral-7b": "Mistral 7b",
    "open-mixtral-8x22b": "Mixtral 8x22b",
    "open-mixtral-8x7b": "Mixtral 8x7b",
    "open-mistral-7b-instruct-v0.1": "Mistral 7b",
    "dbrx-instruct": "DBRX",
    "command-r-plus": "Command R Plus",
    "gemma-7b-it": "Gemma 7b",
    "gemma-2b-it": "Gemma 2b",
    "gemini-1.5-pro-latest": "Gemini 1.5",
    "gemini-pro": "Gemini 1.0",
    "qwen1.5-7b-chat": "Qwen 1.5 7b",
    "qwen1.5-14b-chat": "Qwen 1.5 14b",
    "qwen1.5-32b-chat": "Qwen 1.5 32b",
    "qwen1.5-72b-chat": "Qwen 1.5 72b",
    "qwen1.5-0.5b-chat": "Qwen 1.5 0.5b",
    "qwen1.5-1.8b-chat": "Qwen 1.5 1.8b",
    "qwen2-72b-instruct": "Qwen2 72b",
    "qwen2.5-72b-instruct": "Qwen2.5 72b",
    "codestral-2405": "Codestral",
    "numinamath-7b-tir": "NuminaMath-7b",
    "qwen1.5-4b-chat": "Qwen 1.5 4b",
    "gpt-4-turbo-preview": "GPT-4 Turbo Preview",
}

def get_boxplot_data(df, model_name, prompt_ids):
    models = model_list
    data = []
    df = df[df.prompt_id.isin(prompt_ids)]
    for model in models:
        model_data = get_model_data(df, model)
        data.extend([{
            'model': model,
            'correct': sample
        } for sample in model_data])

    data = pd.DataFrame(data)


    meds = data.groupby('model')['correct'].mean()
    meds.sort_values(ascending=False, inplace=True)
    data = data.set_index('model').join(meds, rsuffix='_mean')

    data = data.reset_index()
    data = data.rename(columns={'correct_mean': 'mean'})
    # data = data[data['mean'] > 0]
    data = data.sort_values(by='mean', ascending=False)
    return data, meds


def plot_boxplot(df, 
                 output_path, 
                 model_list=None, 
                 prompt_ids=None, 
                 title=None,
                fig_width=12,
                fig_height=8,
                 axis_fontsize=10,
                 title_fontsize=12):
    """
    Plot the results of the models.
    Args:
        data (pd.DataFrame): Dataframe with the results.
        output_path (str): Path to save the plot.
    """
    sns.set_style("whitegrid")
    FIG_WIDTH, FIG_HEIGHT = (fig_width, fig_height)

    X_LOWER, X_UPPER = 0, 1
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    data, meds = get_boxplot_data(df, model_list, prompt_ids)

    ax = sns.boxplot(
        data=data,
        x='correct',
        y='model',
        # orient='h',
        showmeans=False,
        shownotches=False,
        showbox=True,
        showcaps=True,
        showfliers=False,
        meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"10"},
        palette='light:#5A9',
        hue='model',
        hue_order=meds.index,
        order=meds.index,
        linewidth=0.3 # make opacity of the boxplot lower if it's zero
    )

    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()

    family = 'sans-serif'

    font = FontProperties()
    font.set_family(family)


    # ax.set_title('Correct answers per model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Correct answers')
    ax.xaxis.tick_top()


    # xlabel = '+' if 91 in prompt_id or  92 in prompt_id  or 93 in prompt_id else ''

    plt.xlabel('AIW Correct response rate' if title is None else title, fontweight='bold', fontdict={ 'size': title_fontsize})

    plt.ylabel('')
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlim(X_LOWER, X_UPPER)
    plt.grid( linestyle='--', alpha=0.5, linewidth=0.1, color='grey')

    plt.tight_layout()
    prompt_ids_str = '_'.join([f'{p}' for p in prompt_ids])

    ax.tick_params(width=0.2)
    ax.figure.savefig(output_path.format(prompt_ids_str), dpi=300, transparent=True)


def do_boxplot(boxplot_data, meds, palette='light:#5A9', hue='model', label=None, color=None):
    if color is not None:
        palette = None
    ax = sns.boxplot(
        data=boxplot_data,
        x='correct',
        y='model',
        showmeans=False,
        shownotches=False,
        showbox=True,
        showcaps=True,
        showfliers=False,
        meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"10"},
        palette=palette,
        color=color,
        hue=hue,
        hue_order=meds.index,
        order=meds.index,
        linewidth=0.3
    )

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    family = 'sans-serif'
    
    font = FontProperties()

    font.set_family(family)


def compare_boxplots(df, 
                     output_path, 
                     model_list=None, 
                     prompt_ids_first=None,
                    prompt_ids_second=None,
                    axis_fontsize=10,
                    title_fontsize=12,
                    fig_width=12,
                    fig_height=8,
                    title=None,
                    legend=None
                    ):
    
    sns.set_style("whitegrid")
    FIG_WIDTH, FIG_HEIGHT = (fig_width, fig_height)
    X_LOWER, X_UPPER = 0, 1
    boxplot_data_first, meds_first = get_boxplot_data(df, model_list, prompt_ids_first)
    boxplot_data_second, meds_second = get_boxplot_data(df, model_list, prompt_ids_second)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    palette1 = sns.color_palette('light:#5A9')
    palette2 = sns.color_palette('dark:#5A8')
    color1 = palette1.as_hex()[1]
    color2 = palette2.as_hex()[1]

    do_boxplot(boxplot_data_first, meds_first, color=color1, label=f'Prompt: {", ".join(prompt_ids_first)}', hue=None)
    do_boxplot(boxplot_data_second, meds_second, color=color2, label=f'Prompt: {", ".join(prompt_ids_second)}', hue=None)

    label = f"{', '.join(prompt_ids_first)} vs {', '.join(prompt_ids_second)}" if title is None else title

    plt.xlabel(label, fontweight='bold', fontdict={ 'size': title_fontsize})
    plt.ylabel('')
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlim(X_LOWER, X_UPPER)
    plt.grid( linestyle='--', alpha=0.5, linewidth=0.1, color='grey')
    legend_labels = legend.split(',') if legend is not None else [f'Prompt: {", ".join(prompt_ids_first)}', f'Prompt: {", ".join(prompt_ids_second)}']
    legend_labels = [l.strip() for l in legend_labels]

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color1, lw=4, label=legend_labels[0]),
        Line2D([0], [0], color=color2, lw=4, label=legend_labels[1])
    ]

    plt.legend(handles=legend_elements, loc='lower right', fontsize=axis_fontsize)

    
    plt.tight_layout()
    prompt_ids_str = '_'.join([f'{p}' for p in prompt_ids_first]) + '_vs_' + '_'.join([f'{p}' for p in prompt_ids_second])

    plt.savefig(output_path.format(prompt_ids_str), dpi=300, transparent=True)




def plot_multiple_barplot(
        df,
        output_path,
        fig_width=12,
        fig_height=8,
        model_list=None,
        prompt_ids=None,
        show_variation=False,
        title='AIW Correct response rate for diffferent prompt variations',
        title_fontsize=12,
        axis_fontsize=10
        ):
    if model_list is not None:
        model_list = [m.lower().replace('-hf', '') for m in model_list]
        df = df[df.model.isin(model_list)]
        model_list = [m.lower().replace('-hf', '') for m in model_list]

    if prompt_ids is not None:
        df = df[df.prompt_id.isin(prompt_ids)]

    if show_variation:
        df["variation"] = df.description.apply(lambda x: re.findall(r"aiw.*?variation \d+", x.lower())[0])
        df["variation"] = df.variation.apply(lambda x: x.replace("variation ", "v").replace(",", ""))
        prompt_ids_variations_dict = df.groupby('variation')['prompt_id'].unique().to_dict()

        df["variation"] = df.variation.apply(lambda x: f"{x}, {', '.join(prompt_ids_variations_dict[x])}")
    else:
        df["variation"] = df.prompt_id

    # df = df.groupby(['variation', 'model'])['correct'].mean().reset_index()

    df["short_model"] = df.model.apply(lambda x: MODEL_MAPPINGS[x])
  
    order = df["variation"].unique()
    if show_variation:
        order = sorted(order, key=lambda x: int(re.findall(r"v\d+", x)[0].replace("v", "")))
    else:
        order = sorted(order, key=lambda x: int(re.findall(r"\d+", x)[0]))

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        x='short_model',
        y='correct',
        hue='variation',
        data=df,
        kind='bar',
        height=fig_height,
        aspect=fig_width/fig_height,
        palette='Set3',
        col_order=order,
        hue_order=order,
        # errorbar='sd',
        # ci='sd',
        order=[MODEL_MAPPINGS[m] for m in model_list]
        )
    
    plt.ylim(0, 1)
    # padding between the title and the plot
    plt.subplots_adjust(top=0.9)
    plt.title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    g.despine(left=True)
    g.set_axis_labels("", "")
    g.legend.set_title("")

    # set legend font size
    for t in g._legend.texts:
        t.set_text(t.get_text())
        t.set_fontsize(axis_fontsize)

    # tight legend if plot is rectangular
    if fig_width > fig_height:
        g.legend.set_bbox_to_anchor([0.9, 0.5])
    g.set_xticklabels(fontsize=axis_fontsize)
    g.set_yticklabels(fontsize=axis_fontsize)

    y_labels = [str("{:.1f}".format(i)) for i in np.arange(0.0, 1.01, step=0.2)]

    ax = plt.gca()
    ax.set_yticklabels(y_labels)

    plt.savefig(output_path.format("_".join(prompt_ids)), dpi=300, bbox_inches='tight', pad_inches=0.1)

def parametrize_response(resp, right_answer):
    resp = int(resp)
    right_answer = int(right_answer)
    mod = resp % right_answer
    # div = right_answer / resp

    if resp == 0:
        return "0"
    if resp > right_answer and mod > 0:
        return f"M+{mod}"
    elif resp < right_answer and mod > 0:
        mod = right_answer - mod
        return f"M-{mod}"
    elif mod == 0:
        return "M"
    
def eval_str(s):
    if s.isnumeric():
        return int(s)
    if s == "M":
        return 0.01
    s = s.replace("M", "0")
    if "+" in s:
        s = list(map(int, s.split("+")))
        return s[0] + s[1]
    elif "-" in s:
        s = list(map(int, s.split("-")))
        return s[0] - s[1]
    else:
        return -1000


def plot_response_distribution_for_multiple_ids(
    df,
    prompt_ids,
    model_names,
    save_path,
    fig_width,
    fig_height
):
    df = df[df.prompt_id.isin(prompt_ids)]
    df = df[df.model.isin(model_names)]
    model_names = [m for m in model_names if m in df.model.unique()]
    if len(prompt_ids) > 1:
        df["parametrized_answer"] = df[["parsed_answer", "right_answer"]].apply(lambda x: parametrize_response(x[0], x[1]), axis=1)
    else:
        df["parametrized_answer"] = df["parsed_answer"]

    fig, axs = plt.subplots(ncols=len(model_names), figsize=(fig_width, fig_height))
    if len(model_names) == 1:
        axs = [axs]
    for i, model in enumerate(model_names):
        df_model = df[df.model == model][["parametrized_answer", "model", "right_answer"]]

        axs[i].tick_params(axis='both', which='major')
        response = df_model.parametrized_answer.values
        response_count_dict = df_model.parametrized_answer.value_counts(sort=False).to_dict()
        response_count = df_model.parametrized_answer.value_counts(sort=False).values
        correct = "M" if len(prompt_ids) > 1 else df["right_answer"].iloc[0]
        num_responses = len(df_model)
        response_count = response_count / num_responses
        response_count_dict = {k: v/num_responses for k, v in response_count_dict.items()}
        # distinct = sorted(list(set(response)))
        distinct = list(response_count_dict.keys())
        distinct = sorted(distinct, key=lambda x: eval_str(x))
        response_count_dict = {k: response_count_dict[k] for k in distinct}


        if correct not in response_count_dict:
            response_count_dict[correct] = 0
            distinct.append(correct)
            distinct = sorted(distinct, key=lambda x: eval_str(x))


        max_num_distinct = 8
        if len(distinct) > max_num_distinct:
            distinct = distinct[:max_num_distinct]
            response_count_dict = {k: response_count_dict[k] for k in distinct}

        axs[i].bar(distinct, response_count_dict.values(), color='black', alpha=0.3, label="Incorrect")
        axs[i].bar([correct], response_count_dict[correct], color="black", alpha=1, label="Correct")
  
        axs[i].set_title(model, fontsize=10)
        axs[i].set_xticks(distinct)
        axs[i].set_yticks(np.arange(0, 1.1, 0.1))
        axs[i].set_ylim(0, 1)
        axs[i].set_xlim(-1, 8)

    save_path = save_path.format("_".join(prompt_ids))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def get_model_data(df, model_name):
    df = df[df.model == model_name]
    data = df.correct.apply(int).values

    n = len(data)
    if n == 0:
        return np.array([])
    p = data.sum()/n
    n = 1
    mean = n*p
    std = np.square(n*p*(1-p))
    data = np.random.normal(loc=mean, scale=std, size=1000)
    return data


def plot_scatter(
        df, 
        benchmark,  
        output_path, 
        prompt_id,
        fig_width=12,
        fig_height=8
        ):

    sns.set_style("whitegrid")
    df = df[~df[benchmark].isna()]
   
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(111)
    df['diff'] = df[benchmark] - df.correct 



    df = df.sort_values(by='diff', ascending=False)
    
    models = df.model.unique()

    df['model_dummy'] = df['model'].map(dict(zip(models, range(len(models)))))

    texts = []


    for i in range(len(df)):
        model_name = df.model.iloc[i]
        correct = df.correct.iloc[i]
        text = MODEL_MAPPINGS[model_name]
        if correct == 0:
            text = ''
        texts.append(plt.text(df.correct.iloc[i]+0.02, df[benchmark].iloc[i], text, fontsize=8, color='black'))

    ax = sns.scatterplot(
        data=df,
        x='correct', 
        y=benchmark, 
        palette="Set3", 
        s=300,
        hue='model',
        style='model',
        ax=ax,
        )
    
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3,
                    box.width, box.height * 0.7])
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5, alpha=0.5))


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        models, 
        loc='lower right',
        fontsize=8, 
        ncol=3, 
        )
    x = np.linspace(0,1, 100)
    plt.plot(
        x,
        x,
        linestyle='--',
        alpha=.5,
        label='y=x',
        color='black'
        )

    xlabel = '+' if 91 in prompt_id or  92 in prompt_id  or 93 in prompt_id else ''
    plt.xlabel(f'AIW{xlabel} Correct response rate', fontweight='bold', fontdict={ 'size': 10})
    plt.ylabel(benchmark, fontweight='bold', fontdict={ 'size': 10})
    plt.xlim(-0.05, 1)
    plt.ylim(0, 1)
    plt.title(f'AIW Correct answers vs {benchmark}', fontsize=10, fontweight='bold')

    plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='grey')
    
    ext = output_path.split('.')[-1]
    prompt_ids_str = '_'.join([f'{p}' for p in prompt_id])
    plt.savefig(output_path.replace(f'.{ext}', '').format(prompt_ids_str)+f'_{benchmark}.{ext}'.replace(':', '_'), dpi=400, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the results of the models')
    parser.add_argument('--output', type=str, default='boxplot.png', help='Path to the output plot')
    parser.add_argument('--jsons_dir', type=str, default='.', help='Path to the directory with json files')
    parser.add_argument('--prompt_ids', nargs='+', type=str, action='append', help='List of prompt ids or list of lists of prompt ids')
    parser.add_argument('--prompts_json', type=str, default='prompts.json', help='Path to the prompts json file')
    parser.add_argument('--model_list_path', type=str, default=None, help='Path to the file with the list of models')
    parser.add_argument('--plot_multiple', action='store_true', help='Plot multiple barplots')
    parser.add_argument('--plot_boxplot', action='store_true', help='Plot boxplot')
    parser.add_argument('--fig_width', type=int, default=12, help='Width of the figure')
    parser.add_argument('--fig_height', type=int, default=8, help='Height of the figure')
    parser.add_argument('--show_variation', action='store_true', help='Show prompt variation')
    parser.add_argument('--title', nargs='?', const='', help='Title of the plot')
    parser.add_argument('--title_fontsize', type=int, default=12, help='Title fontsize')
    parser.add_argument('--axis_fontsize', type=int, default=10, help='Axis fontsize')
    parser.add_argument('--compare_boxplots', action='store_true', help='Compare boxplots')
    parser.add_argument('--boxplot_legend', type=str, default="Type1,Type2", help='Legend for the boxplot')
    parser.add_argument('--plot_response_distribution', action='store_true', help='Plot response distribution')
    parser.add_argument('--plot_scatter', action='store_true', help='Plot scatter')


    args = parser.parse_args()


    dist = stats.binom


    jsons_dir = args.jsons_dir
    prompt_ids = args.prompt_ids
    prompts_json = args.prompts_json
    models_list_path = args.model_list_path
    if models_list_path is not None:
        with open(models_list_path, 'r') as f:
            models_list = json.load(f)
            model_list = [m.lower().replace('-hf', '') for m in models_list]
        
    else:
        models_list = None
    

    with open(prompts_json, 'r') as f:
        prompts = json.load(f)

    prompts = {p["prompt"]: p for p in prompts if isinstance(p["prompt"], str)}


  
    data = []


    for j in glob.glob(f'{jsons_dir}/**/*.json*', recursive=True):
        if j.endswith('.gz'):
            js = pd.read_json(j, compression='gzip')
            data.extend(js.to_dict(orient='records'))
        else:
            with open(j, 'r') as f:
                j_str = f.read()
                try:
                    js = json.loads(j_str)
                    if isinstance(js, list):
                        data.extend(js)
                    else:
                        data.append(js)
                except Exception as e:
                
                    if '}{' in j_str:
                        lines = j_str.replace('}{', '}\n\n{').split('\n\n')
                    else:
                        lines = j_str.replace('}\n{', '}\n\n{').split('\n\n')

                    lines = [line for line in lines if line]
                    
                    for line in lines:
                        try:
                            js = json.loads(line)
                            data.append(js)
                        except Exception as e:
                            continue
   

 
    df = pd.DataFrame(data)
    df = df.dropna(subset=['prompt'])
    df.prompt = df.prompt.apply(lambda x: x.strip().replace("\\'", "'" ))
    additional_format = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags. <|eot_id|><|start_header_id|>user<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"]
    # remove additional format from the prompt
    df.prompt = df[["model", "prompt"]].apply(lambda x: x[1].replace(additional_format[0], "").replace(additional_format[1], "").strip() if "reflection" in x[0].lower() else x[1], axis=1)

    df['prompt_id'] = df.prompt.apply(lambda x: prompts.get(x, {}).get('id', None))
    
    df = df.dropna(subset=['prompt_id'])
    
    df.model = df.model.apply(lambda x: x.split('/')[-1].lower().replace("-hf", ""))
    # df.model = df.model.apply(lambda x: x.replace("instruct", 'chat') if 'llama' in x else x)
    
    df.prompt_id = df.prompt_id.astype(int).astype(str)
    df["description"] = df.prompt.apply(lambda x: prompts.get(x, {}).get('description', None))
    if len(prompt_ids) == 1:
        prompt_ids = prompt_ids[0]

    
    
    if args.plot_multiple:
        plot_multiple_barplot(
            df, 
            args.output, 
            args.fig_width, 
            args.fig_height, 
            models_list, 
            prompt_ids, 
            args.show_variation, 
            args.title,
            title_fontsize=args.title_fontsize,
            axis_fontsize=args.axis_fontsize
            )
        

    
    if args.plot_boxplot:
        plot_boxplot(
            df, 
            args.output, 
            model_list, 
            prompt_ids,
            args.title,
            args.fig_width,
            args.fig_height,
            args.axis_fontsize,
            args.title_fontsize
            )

    if args.compare_boxplots:
        prompt_ids_first = prompt_ids[0]
        prompt_ids_second = prompt_ids[1]
        compare_boxplots(
            df, 
            args.output, 
            model_list, 
            prompt_ids_first,
            prompt_ids_second,
            args.axis_fontsize,
            args.title_fontsize,
            args.fig_width,
            args.fig_height,
            args.title,
            args.boxplot_legend
            )
        
    if args.plot_response_distribution:
        if len(prompt_ids) > 1:
            for prompt_id in prompt_ids:
                plot_response_distribution_for_multiple_ids(
                    df,
                    [prompt_id],
                    model_list,
                    args.output,
                    args.fig_width,
                    args.fig_height
                )
        else:
            plot_response_distribution_for_multiple_ids(
                df,
                prompt_ids,
                model_list,
                args.output,
                args.fig_width,
                args.fig_height
            )

    if args.plot_scatter:
        df = df.groupby(['model'])[['correct']].mean().reset_index()

        df_names = pd.read_excel('models_data/models_plot_set_FULL_names.xlsx')

        df_lb = pd.read_excel('models_data/all_models_plot_set_FULL_results.xlsx')
        df_lb.model = df_lb.model.apply(lambda x: x.replace('-hf', ''))
        df_names.model = df_names.model.apply(lambda x: x.replace('-hf', ''))
        df_names.hf_model_name  = df_names.hf_model_name.apply(lambda x: x.replace('-hf', ''))
        columns = ['hf_model_name'] + df_lb.columns[1:].tolist()
        df_lb.columns = columns

        df_lb.hf_model_name = df_lb.hf_model_name.str.lower()
        df_names.hf_model_name = df_names.hf_model_name.str.lower()
        df_lb = df_lb.merge(df_names, left_on='hf_model_name', right_on='hf_model_name', how='right')
        df_lb.model = df_lb.model.apply(lambda x: x.lower().split('/')[-1].split('\/')[-1])
        
        benchmark = 'MMLU'

        df.model = df.model.apply(lambda x: x.lower()).astype(str)
        df = df.set_index('model').join(df_lb.set_index('model'))

        
        df = df[['correct', benchmark]].reset_index()
    
    
        plot_scatter(
            df, 
            benchmark, 
            args.output, 
            prompt_ids,
            args.fig_width,
            args.fig_height
            )