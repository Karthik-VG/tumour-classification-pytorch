import os
import logging
from numpy import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import argparse
from torchvision impot models
from src.utils.common import read_yaml, create_directories

STAGE = "stage_02_base_model_creation"

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a+"
                    )


class CNN:
  def model(self):
    model=models.resnet50(pretrained=True)
    model.classifier = nn.Sequential(
    nn.Linear(in_features=9216, out_features=100, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=100, out_features=4, bias=True)
    )
    return model

if __name__ == "__main__":
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--config", '-c', default="config/config.yaml")
        parsed_args = arg_parser.parse_args()
        content  = read_yaml(parsed_args.config)
        model_path = os.path.join(content['artifacts']['model'])
        create_directories([model_path])
        model_name = content['artifacts']['base_model']
        full_model_path = os.path.join(model_path, model_name)
        model_ob = CNN().model()
        torch.save(model_ob, full_model_path)
        logging.info(f"model created and saved at {full_model_path}")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        logging.exception(e)
        raise e