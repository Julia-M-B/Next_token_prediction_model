import os
import warnings
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_conversation_url(conversation_id: str, limit: int = 10000) -> str:
    return f"https://spokes.clarin-pl.eu/restapi/download/full_text/excel?id=%22%22&text_id={conversation_id}&limit={limit}&offset=0&filter=&orderBy=seq&orderDir=asc"


def download_conversations_transcripts(
    conversations_df: pd.DataFrame, dir_path: str, dataset_name: str
):
    dataset_path = Path(f"{dir_path}/{dataset_name}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    for c_id in tqdm(conversations_df["Id"].values):
        response = requests.get(get_conversation_url(c_id))
        c_path = str(dataset_path) + f"/{c_id}.xls"
        with open(c_path, "wb") as f:
            f.write(response.content)
        yield c_path


def change_xls_to_txt(file_path: str):
    df = pd.read_excel(file_path)
    utterances = [utt for utt in df["Utt"].values if utt and isinstance(utt, str)]
    lines = "\n".join(utterances)
    with open(f"{file_path[:-4]}.txt", "w") as f_out:
        f_out.write(lines)
    os.remove(file_path)


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    conversations = pd.read_excel(config["conversations-id-file"])
    conversations_train = conversations[conversations["Id"].str.startswith("CBIZ")]
    conversations_test = conversations[~conversations["Id"].str.startswith("CBIZ")]

    print("Saving train conversations ... ")
    for file_path in download_conversations_transcripts(
        conversations_df=conversations_train,
        dir_path=config["dialog-dir-path"],
        dataset_name=config["conversations-train-dir-name"],
    ):
        change_xls_to_txt(file_path)

    print("Saving test conversations ... ")
    for file_path in download_conversations_transcripts(
        conversations_df=conversations_test,
        dir_path=config["dialog-dir-path"],
        dataset_name=config["conversations-test-dir-name"],
    ):
        change_xls_to_txt(file_path)
