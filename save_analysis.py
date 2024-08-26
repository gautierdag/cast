import wandb
import json
import copy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_runs_from_wandb(project, entity):

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    dfs = []
    # download all tables
    for run in runs:
        artifacts = run.logged_artifacts()
        for artifact in artifacts:
            print(artifact.name)
            table_dir = artifact.download()
            table_path = f"{table_dir}/evaluated_statements.table.json"
            with open(table_path) as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            df["model"] = run.name
            dfs.append(df)

    df = pd.concat(dfs)
    return df


def get_validator(col_name: str):
    positive, negative = col_name.split("_")[-1].split("|")

    def parse_validator(x):
        if x.lower().startswith(positive):
            return 1
        elif x.lower().startswith(negative):
            return 0
        else:
            # print(x)
            return 0

    return parse_validator


def get_label_specific_dfs(filterd_master_df):
    sub_dfs = {}
    for  key, pos_neg in {"both": "both|one", "true":"true|false", "yes":"yes|no"}.items():

        df = copy.deepcopy(filterd_master_df)
        # df.columns
        validate_columns = [col for col in df.columns if "validate_" in col and pos_neg in col]

        df.dropna(subset=validate_columns, inplace=True)
        for col in validate_columns:
            df[col] = df[col].apply(get_validator(col))

        text_validate_columns = [
            col for col in df.columns if "text" in col and "validate_" in col and pos_neg in col
        ]
        image_validate_columns = [
            col for col in df.columns if "image" in col and "validate_" in col and pos_neg in col
        ]
        both_validate_columns = [
            col for col in df.columns if "both" in col and "validate_" in col and pos_neg in col
        ]

        drop_validate_columns = [
            col for col in df.columns if "validate_" in col and pos_neg not in col
        ]

        # df.dropna(subset=validate_columns, inplace=True)

        # aggregate
        df["validate_text_avg"] = df[text_validate_columns].mean(axis=1)
        df["validate_image_avg"] = df[image_validate_columns].mean(axis=1)
        df["validate_both_avg"] = df[both_validate_columns].mean(axis=1)

        ## drop columns
        df.drop(columns=drop_validate_columns, inplace=True)
        sub_dfs[key] = copy.deepcopy(df)
    
    return sub_dfs


def get_merged_metrics_df(sub_dfs, statement_idx):
    agg_dfs = {}
    for key, _df in sub_dfs.items():
        df_agg = _df[_df["generated_idx"] <= statement_idx].groupby(["model", "generated_with"]).agg(
                {
                    "validate_text_avg": "mean",
                    "validate_image_avg": "mean",
                    "validate_both_avg": "mean",
                }
            )

        agg_dfs[key] = copy.deepcopy(df_agg)
    
    ## merge the dataframes
    df_merged = agg_dfs["both"].merge(agg_dfs["yes"], on=["model", "generated_with"], suffixes=("_both", "_yes"))
    df_merged = df_merged.merge(agg_dfs["true"], on=["model", "generated_with"], suffixes=("_both", "_true"))

    ## Add average column for text, image and both
    df_merged["validate_text_avg_avg"] = df_merged[["validate_text_avg_yes", "validate_text_avg_both", "validate_text_avg"]].mean(axis=1)
    df_merged["validate_image_avg_avg"] = df_merged[["validate_image_avg_yes", "validate_image_avg_both", "validate_image_avg"]].mean(axis=1)
    df_merged["validate_both_avg_avg"] = df_merged[["validate_both_avg_yes", "validate_both_avg_both", "validate_both_avg"]].mean(axis=1)

    ## reorder the columns
    df_merged = df_merged[["validate_text_avg_yes", "validate_text_avg_both", "validate_text_avg", "validate_text_avg_avg",
                        "validate_image_avg_yes", "validate_image_avg_both", "validate_image_avg", "validate_image_avg_avg",
                        "validate_both_avg_yes", "validate_both_avg_both", "validate_both_avg", "validate_both_avg_avg"
                        ]]

    with open(f"/home/multimodal-self-consistency/plots/results_{statement_idx}.tex", 'w') as tf:
        tf.write(df_merged.to_latex(float_format="%.2f"))

    dropcolumns = ["validate_text_avg_yes", "validate_text_avg_both", "validate_text_avg", "validate_image_avg_yes", "validate_image_avg_both", "validate_image_avg",
            "validate_both_avg_yes", "validate_both_avg_both", "validate_both_avg"]

    df_merged.drop(columns=dropcolumns, inplace=True)
    df_merged.reset_index(inplace=True)
    df_agg1 = df_merged[df_merged["generated_with"] == "text"][["model", "generated_with", "validate_text_avg_avg"]]
    df_agg2 = df_merged[df_merged["generated_with"] == "image"][["model", "generated_with", "validate_image_avg_avg"]]
    df_agg3 = df_merged[df_merged["generated_with"] == "both"][["model", "generated_with", "validate_both_avg_avg"]]


    ## merge model with generated_with and model
    df1 = df_agg1.merge(df_agg2, on="model", suffixes=("", "_image"))
    df2 = df1.merge(df_agg3, on="model", suffixes=("", "_both"))

    return df2


def save_heatmap(df2, statement_idx):
    _df2 = df2.copy()
    _df2.rename(columns={"validate_text_avg_avg": "text", "validate_image_avg_avg": "image", "validate_both_avg_avg": "both"}, inplace=True)

    _df2 = _df2[["model", "text", "image", "both"]]
    _df2.set_index("model", inplace=True)

    _df2["text"] = _df2["text"].astype(float)
    _df2["image"] = _df2["image"].astype(float)
    _df2["both"] = _df2["both"].astype(float)

    # Plot heat map for df2
    plt.figure(figsize=(12, 12))
    sns_plot = sns.heatmap(_df2, annot=True, cmap="coolwarm", fmt=".2f")
    # plt.title("Consistency")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ## increase font size of the values in the heatmap
    for text in sns_plot.texts:
        text.set_fontsize(28)

    ## increase font size of the color bar
    cbar = sns_plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    ## save sns plot
    sns_plot.figure.savefig(f"/home/multimodal-self-consistency/plots/consistency_{statement_idx+1}_statements.png", bbox_inches="tight")



if __name__ == "__main__":
    project = "consistency"
    entity = "itl"
    our_models = {
        "Bunny": "bunny-all-validate-prompts",
        "LLaVA": "llava-all-validate-prompts",
        "MiniCPM": "minicpm2-minicpm-all-validate-prompts",
        "InternVL2": "internvl-all-validate-prompts",
        "GPT-4o-Mini": "gpt4o-mini-"
    }

    master_df = get_runs_from_wandb(project, entity)

    consistency_models_df = master_df[master_df.model.isin(our_models.values())]
    consistency_models_df["model"] = consistency_models_df["model"].map({v: k for k, v in our_models.items()})

    print(consistency_models_df.model.unique())

    for statement_idx in range(4):
        print(f"Statement: {statement_idx+1}")
        statement_df = consistency_models_df[consistency_models_df.generated_idx <= statement_idx]
        
        sub_dfs = get_label_specific_dfs(statement_df)

        _merged_metrics_df = get_merged_metrics_df(sub_dfs, statement_idx)
        save_heatmap(_merged_metrics_df, statement_idx)
