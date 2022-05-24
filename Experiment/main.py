import argparse
import re
import nn_runner
import warnings

from data_process import DataProcessor

import sys

# Data downloaded from here: https://github.com/MiguelAMartinez/flowers-image-classifier/blob/master/image_classifier.ipynb
# https://github.com/Muhammad-MujtabaSaeed/102-Flowers-Classification/blob/master/102_Flowers_classification.ipynb
# {'test': 819, 'train': 6552, 'valid': 818} / 10%-80%-10%

# images folder was replaced by https://github.com/ml4py/dataset-iiit-pet after download.
# {'test': 739, 'train': 5912, 'valid': 739} / 10%-80%-10%

def report(args_dict):
    try:
        with open("output.txt", 'a') as F:
            F.write("\n" + 40 * "=" + " REPORT " + 40 * "=" + "\n")
            F.write("dataset: {} \n\n".format(args_dict["data_dir"]))
            F.write("model: {} \n\n".format(args_dict["model"]))
            F.write("noise_type: {} \n\n".format(args_dict["noise_type"]))
            F.write(args_dict["processor"].T_Noise.__str__())
            losses = [loss.detach().cpu().numpy().tolist() for loss in args_dict["batch"]["loss"]]
            F.write("loss per batch: {} \n\n".format(losses))
            F.write("accuracy per batch: {}\n\n".format(args_dict["batch"]["accuracy"]))
            F.write("Final Loss: {}\n\n".format(args_dict["loss"]))
            F.write("Final accuracy: {}\n\n".format(args_dict["acc"]))
            F.write("Total time (minutes): {}\n\n".format(args_dict["total_time"])) 
            F.write("Epochs: {}\n\n".format(args_dict["epochs"]))           
    except IOError as error:
        print("IOError: {}".format(error))   
        

def parse_cfg(filename):
    args = {}
    try:
        with open(filename, 'r') as F:
            while True:    
                line = F.readline()
                if not line:
                    break
                
                if '#' in line or len(line.strip()) in range(0, 2):
                    continue

                parts = re.split('\s+|=', line)
                parts = [p for p in parts if p != '']
                if len(parts) != 2:
                    print("Skipping incompatible config line: {} --> Cause: too many arguments provided.".format(line))
                    continue
                
                key = parts[0]
                value = parts[1]
                
                if key == "amount" \
                    or key == "mean" \
                    or key == "std" \
                    or key == "init_erasing_prob" \
                    or key == "epsilon" \
                    or key == "alpha":
                    args[key] = float(value)
                elif key == "attempts" \
                    or key == "subarea_low" \
                    or key == "subarea_high" \
                    or key == "aspect_ratio_low" \
                    or key == "aspect_ratio_high" \
                    or key == "epochs" \
                    or key == "iters":
                    args[key] = int(value)
                else:
                    args[key] = value
    except IOError as error:
        print("IOError: {}".format(error))
        sys.exit(-1)
            
    return args
            

def main():
    # Useful links
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html#torchvision.datasets.Flowers102
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html#torchvision.datasets.OxfordIIITPet
    # https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/pytorch_semantic_segmentation.ipynb#scrollTo=mIsVdt7qSDyU

    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    
    args = parser.parse_args()
    args = parse_cfg(args.cfg)
    
    processor = DataProcessor(args=args)
        
    args["processor"] = processor
         
    train_loader, valid_loader, test_loader = processor.load_transform()
    
    loss, acc, batch, total_time = nn_runner.run (
        train_loader, 
        valid_loader, 
        test_loader, 
        args=args
    )
    
    args["loss"] = loss
    args["acc"] = acc
    args["batch"] = batch
    args["total_time"] = total_time    

    report(args)
    

if __name__ == "__main__":
    main()
    

