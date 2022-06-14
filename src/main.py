from clearml import Task, Dataset
import sys
import os
import yaml
import argparse
import itertools

import train_aip

PROJECT_NAME = 'scaled_yolo_v4'

def get_args():
    # Exact from train.py
    parser = argparse.ArgumentParser()
    #ClearML Bits
    parser.add_argument('--clearml', action='store_true', help='Log Experiment with ClearML')
    parser.add_argument('--remote', action='store_true', help='Remote Execution')
    parser.add_argument('--s3', action='store_true', help='Retrieve Training Data from S3 instead')
    parser.add_argument('--data_proj_name', type=str, default='', help='dataset_project arg for ClearML Dataset')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = get_args() #Get Clearml args

    train_args = {'weights': 'yolov4-p5.pt',
                'cfg': '',
                'data': 'data/goog_dummy.yaml',
                'hyp': '',
                'epochs': 1,
                'batch_size': 8,
                'img_size': [640, 640],
                'rect': False,
                'resume': False,
                'nosave': True,
                'notest': False,
                'noautoanchor': False,
                'evolve': False,
                'bucket': '',
                'cache_images': False,
                'name': 'test',
                'device': 0,
                'multi_scale': False,
                'single_cls': False,
                'adam': False,
                'sync_bn': False,
                'local_rank': -1,
                'logdir': 'runs/'
    }

    clearml_task = Task.init(project_name=PROJECT_NAME, task_name='syolov4_'+train_args.name)

    ## Set Base Docker & bits for remote execution
    if args.s3:
        # Obtain Dataset 
        if args.data_proj_name=='':
            print("No data project name provided! Terminating...")
            exit()
        # If loading from S3, overwrite the train/val/test path with ClearML get_local_copy path
        with open(train_args.data, 'r') as f:
            data_yaml = yaml.safe_load(f)
        train_path = Dataset.get(dataset_name='train', dataset_project=args.data_proj_name).get_local_copy() # Get Train Data
        train_path = os.path.join(train_path, '')
        val_path = Dataset.get(dataset_name='val', dataset_project=args.data_proj_name).get_local_copy() # Get Val Data
        val_path = os.path.join(val_path, '')
        test_path = Dataset.get(dataset_name='test', dataset_project=args.data_proj_name).get_local_copy() # Get Test Data
        test_path = os.path.join(test_path, '')
        data_yaml['train']=train_path
        data_yaml['val']=val_path
        data_yaml['test']=test_path
        new_yaml_file = args.data[:-5]+"_m.yaml" if args.data[:-5]=='.yaml' else args.data[:-4]+"_m.yaml"
        with open(new_yaml_file, 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)
        train_args.data = new_yaml_file # use the modified yaml file

    args_list = []
    for x in train_args:
        args_list.append('--'+x)
        args_list.append(train_args[x])

    train_aip.main(args_list)

    ## To verify whether clearml is able to capture training through this means
    ## To return path to saved model from train_aip in local path for uploading, if s3=True
