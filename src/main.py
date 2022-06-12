from clearml import Task, Dataset
import sys
import os
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4-p5.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')

    #ClearML Bits
    parser.add_argument('--clearml', action='store_true', help='Log Experiment with ClearML')
    parser.add_argument('--remote', action='store_true', help='Remote Execution')
    parser.add_argument('--s3', action='store_true', help='Retrieve Training Data from S3 instead')
    parser.add_argument('--data_proj_name', type=str, default='', help='dataset_project arg for ClearML Dataset')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = get_args()

    ## Init Task
    ## Set Base Docker

    # Obtain Dataset 
    if args.data_proj_name=='':
        print("No data project name provided! Terminating...")
        exit()
    # If loading from S3, overwrite the train/val/test path with ClearML get_local_copy path
    print(os.listdir())
    with open(args.data, 'r') as f:
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
    args.data = new_yaml_file # use the modified yaml file

    # Init Train Class