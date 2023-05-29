import time
import json
from donut import DonutFinetuning
from data_loader import load_sroie_dataset
from sagemaker.huggingface import HuggingFace

# Initialization
dataset = load_sroie_dataset()
finetuning = DonutFinetuning()
preprocessed_dataset = finetuning.preprocess_training_dataset(dataset)

# define Training Job Name
job_name = f'huggingface-donut-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# stingify special tokens
special_tokens = ",".join(finetuning.processor.tokenizer.special_tokens_map_extended["additional_special_tokens"])

# hyperparameters, which are passed into the training job
hyperparameters = {
  'model_id': finetuning.model_id,                      # pre-trained model
  'special_tokens': json.dumps(special_tokens),        # special tokens which will be added to the tokenizer
  'dataset_path': '/opt/ml/input/data/training',       # path where sagemaker will save training dataset
  'epochs': 3,                                         # number of training epochs
  'per_device_train_batch_size': 8,                    # batch size for training
  'gradient_checkpointing': True,                      # batch size for training
  'lr': 4e-5,                                          # learning rate used during training
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'train.py',        # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = 'ml.g5.2xlarge',   # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 100,               # the size of the EBS volume in GB
    transformers_version = '4.26',            # the transformers version used in the training job
    pytorch_version      = '1.13',            # the pytorch_version version used in the training job
    py_version           = 'py39',            # the python version used in the training job
    hyperparameters      =  hyperparameters
)

# define a data input dictonary with our uploaded s3 uris
# data = {'training': training_input_path}
data = preprocessed_dataset["train"]

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)
