from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def download_hf_data(url: str, dir_path: str, dataset_name: str):
    """
    Downloads huggingface news dataset form huggingface hub.
    """
    df = pd.read_csv(url)

    dataset_path = Path(f"{dir_path}/{dataset_name}")
    dataset_path.mkdir(parents=True, exist_ok=True)

    for i, text in tqdm(enumerate(df["content"].values, 1)):
        file_path = dataset_path / f"news_{i}.txt"
        if type(text) == str:
            with open(file_path, "w") as f_out:
                f_out.write(text)


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    download_hf_data(
        url=config["huggingface-dataset-url"],
        dir_path=config["dir-path"],
        dataset_name=config["huggingface-dataset-name"],
    )
