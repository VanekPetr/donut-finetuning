from datasets import load_dataset
import json
from pathlib import Path
import shutil


def load_sroie_dataset():
    # define paths
    base_path = Path("data")
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    # define metadata list
    metadata_list = []

    # parse metadata
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            # load json file
            data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(data)
            # add to metadata list if image exists
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append({"text":text,"file_name":f"{file_name.stem}.jpg"})
                # delete json file

    # write jsonline file
    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    # remove old meta data
    shutil.rmtree(metadata_path)

    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

    print(f"Dataset has {len(dataset)} images")
    print(f"Dataset features are: {dataset.features.keys()}")

    return dataset


def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj
