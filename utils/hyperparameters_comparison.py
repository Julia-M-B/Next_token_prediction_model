import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl

"""
Notatki:

Skrypt służący do analizy wpływu hiperparametrów na "metrykę decyzyjną",
tj. top5 accuracy (czy kolejne słowa pojawiło się wśród 5 najlepszych
kandydatów na kolejne słowo). 

Porównania muszą być wewnątrz-study (tj. jeśli dany hiperparametr był dobierany
w study nr 1, to tylko wyniki ze study nr 1 są brane pod uwagę).

Parametry do porównania:
capacity: tokenizer, emb_dim
LSTM: hidden_dim, n_layers
regularization: weight_decay, dropout
learning: batch_size, seq_len, lr

Z pola trial params wyciągamy, które parametry były analizowane w ramach tego 
study.

Ze study wyciągamy numer study i nr triala

 - Dla każdego analizowanego parametru wewnątrz study robimy sprawdzenie 
średniego względnego wzrostu procentowego i zapisujemy je do 
wspólnej dla wszystkich parametrów tabeli z wynikami. (9 wykresów)

 - Rysujemy historię top 5 konfiguracji (4 wykresy)

 - Dla study 1-3 robimy korelację między val perplexity a val accuracy (1 wykres)
"""

# set style
sns.set_theme(
    context="paper",
    style="whitegrid"
)

def load_single_study_results(
        study_num: int,
        results_dir: str = "next_token_prediction_model/study_results"
):
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob(f"study_v2_{study_num}_*.json"))

    rows = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        row = {
            "study_num": study_num,
            "best_val_acc5": data['best_score'],
            # "best_val_acc5": data['best_val_acc5'],
        }
        for param, value in data['params'].items():
            row[param] = value

        for i, trial_param in enumerate(data["trial_params"]):
            row[f"trial_param_{i}"] = trial_param

        rows.append(row)

    return pd.DataFrame(rows)

def calculate_param_stats(df: pd.DataFrame, param_name: str) -> pd.DataFrame | None:
    study_df = df.copy()
    if study_df[param_name].nunique() <= 1:
        return
    param_stats = study_df.groupby(param_name)["best_val_acc5"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    return param_stats.sort_values('max', ascending=False)

def plot_bar_chart_with_error_bars(param_name: str, param_stats: pd.DataFrame, plot_names_mapping: dict, output_dir: str):
    plot_name = plot_names_mapping.get(param_name) if plot_names_mapping.get(param_name) else param_name
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = range(len(param_stats))
    means = param_stats['mean'].values
    stds = param_stats['std'].values
    bars = ax.bar(x, means, color=mpl.color_sequences["Set2"] , yerr=stds, capsize=10, alpha=0.7)
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.002,
                 f'{mean:.4f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel("Rozważane wartości parametru", fontsize=14, fontweight='bold')
    ax.set_ylabel("Największa dokładność (top 5) modelu", fontsize=14,
                   fontweight='bold')
    ax.set_title(plot_name, fontsize=16, fontweight='bold')
    ax.set_xticks(x)

    if param_name == "tokenizer":
        labels = [l.replace('spm_', '').replace('.model', '') for l in
                  param_stats.index]
    else:
        labels = param_stats.index

    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{param_name}_analysis.png", dpi=300,
                bbox_inches='tight')
    plt.close()

def main():

    PARAMS_NAMES_MAPPING = {
        "tokenizer": "Tokenizer (rozmiar słownika)",
        "emb_dim": "Wymiar warstwy zagnieżdzającej",
        "n_layers": "Liczba warstw LSTM",
        "hidden_units": "Wymiar stanu ukrytego warstw LSTM",
        "dropout": "Współczynnik porzucenia neuronów (dropout)",
        "weight_decay": "Zanik wag (weight decay)",
        "batch_size": "Wielkość paczki",
        "seq_len": "Długość sekwencji",
        "lr": "Współczynnik uczenia"
    }

    # Configuration
    RESULTS_DIR = "next_token_prediction_model/study_results"
    OUTPUT_DIR = "hyperparameter_analysis"

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for study_num in range(4):
        study_df = load_single_study_results(study_num=study_num)
        trial_columns = [col for col in study_df.columns.values.tolist() if col.startswith("trial_param_")]
        for trial_column in trial_columns:
            param_name = study_df[trial_column][0]
            param_stats = calculate_param_stats(
                df=study_df,
                param_name=param_name
            )
            plot_bar_chart_with_error_bars(
                param_name=param_name,
                param_stats=param_stats,
                plot_names_mapping=PARAMS_NAMES_MAPPING,
                output_dir=OUTPUT_DIR
            )
            print(param_stats)


if __name__ == "__main__":
    main()