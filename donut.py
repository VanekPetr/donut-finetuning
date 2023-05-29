import json
from transformers import DonutProcessor
from data_loader import json2token, load_sroie_dataset


class DonutFinetuning(object):

    def __init__(self):
        # Load processor
        self.model_id = "naver-clova-ix/donut-base"
        self.processor = DonutProcessor.from_pretrained(self.model_id)
        self.new_special_tokens = []  # new tokens which will be added to the tokenizer
        self.task_start_token = "<s>"  # start of task token
        self.eos_token = "</s>"  # eos token of tokenizer
        # we update some settings which differ from pretraining; namely the size of the images + no rotation required
        # resizing the image to smaller sizes from [1920, 2560] to [960,1280]
        self.processor.feature_extractor.size = [720, 960]  # should be (width, height)
        self.processor.feature_extractor.do_align_long_axis = False

    def preprocess_documents_for_donut(self, sample):
        # create Donut-style input
        text = json.loads(sample["text"])
        d_doc = self.task_start_token + json2token(text) + self.eos_token
        # convert all images to RGB
        image = sample["image"].convert('RGB')
        return {"image": image, "text": d_doc}

    def transform_and_tokenize(self, sample, split="train", max_length=512, ignore_id=-100):
        # create tensor from image
        try:
            pixel_values = self.processor(
                sample["image"], random_padding=split == "train", return_tensors="pt"
            ).pixel_values.squeeze()
        except Exception as e:
            print(sample)
            print(f"Error: {e}")
            return {}

        # tokenize document
        input_ids = self.processor.tokenizer(
            sample["text"],
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token

        return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

    def preprocess_training_dataset(self, dataset):
        proc_dataset = dataset.map(self.preprocess_documents_for_donut)

        print(f"Sample: {proc_dataset[45]['text']}")
        print(f"New special tokens: {self.new_special_tokens + [self.task_start_token] + [self.eos_token]}")
        # add new special tokens to tokenizer
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.new_special_tokens + [self.task_start_token] + [self.eos_token]})

        # need at least 32-64GB of RAM to run this
        processed_dataset = proc_dataset.map(self.transform_and_tokenize, remove_columns=["image", "text"])

        processed_dataset = processed_dataset.train_test_split(test_size=0.1)

        return processed_dataset
