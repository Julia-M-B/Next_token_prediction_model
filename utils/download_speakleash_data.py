from pathlib import Path

import yaml
from speakleash import Speakleash


def save_quality_docs(
    sl: Speakleash,
    dir_path: str,
    datasets_names: list[str],
    limit: int = 0,
    quality: list[str] = ["high"],
):
    """
    Saves text documents from Speakleash dataset to a given directory.
    """
    for dataset_name in datasets_names:
        path = Path(f"{dir_path}/{dataset_name}")
        path.mkdir(parents=True, exist_ok=True)
        ds = sl.get(dataset_name).ext_data
        counter = 0
        if not limit:
            limit = 1_000_000
        for doc in ds:
            txt, meta = doc
            q = meta.get("quality", "").lower()
            if q in quality:
                with open(
                    path.joinpath(f"{dataset_name}_{counter + 1}.txt"),
                    "w",
                    encoding="utf-8",
                ) as out_file:
                    out_file.write(txt)
                counter += 1
                if counter == limit:
                    break
        print(f"Saved {counter} documents to the {path} directory.")


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    sl = Speakleash(config["dir-path"])
    save_quality_docs(
        sl,
        dir_path=config["dir-path"],
        datasets_names=config["speakleash-datasets"],
        limit=config["data-limit"],
    )
