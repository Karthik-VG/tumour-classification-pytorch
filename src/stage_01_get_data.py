import logging
import os
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.utils.common import create_directories, read_yaml


STAGE = "stage_01_get_data"

log_file = os.path.join("logs", 'running_logs.log')


logging.basicConfig(filename=log_file, level=logging.INFO,
                    format= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode='a'

                    )

def main(config_path):
    content = read_yaml(config_path)
    data_folder_path = content['Data']['root_data_folder']
    create_directories([data_folder_path])
    logging.info(f"getting data")

    train_tranforms = transforms.Compose([
                                      transforms.Resize(content["params"]["IMAGE_SIZE"]),
                                      transforms.RandomRotation(degrees=20),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
    ])

    test_tranforms = transforms.Compose([
                                      transforms.Resize(content["params"]["IMAGE_SIZE"]),
                                      transforms.ToTensor(),

    ])
    
    train_path = Path("Training")
    
    test_path = Path("Testing")
    
    train_data =  datasets.ImageFolder(root=train_path, transform=train_tranforms)

    test_data = datasets.ImageFolder(root=test_path, transform=test_tranforms)
    
    given_label = train_data.class_to_idx

    label_map = {val: key for key, val in given_label.items()}

    logging.info(f"data is available at {data_folder_path}")

    logging.info(f"getting dataloader")

    train_data_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=content['params']['BATCH_SIZE'])
    
    test_data_loader = DataLoader(dataset=test_data, batch_size=content['params']['BATCH_SIZE'], shuffle=False)

    return train_data_loader, test_data_loader, label_map



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e