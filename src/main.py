from threading import local
from clearml import Task, Dataset
import os
import yaml
import argparse
import subprocess


PROJECT_NAME = 'scaled_yolo_v4'

def generate_arg_dict():
    parser = argparse.ArgumentParser()
    # PASTE ALL `add_argument` LINES HERE
    opt = parser.parse_args(args=[])
    return vars(opt)


def get_args():
    # Exact from train.py
    parser = argparse.ArgumentParser()
    #ClearML Bits
    parser.add_argument('--train_config', type=str, default='', help='path to yaml file of training configuration')
    parser.add_argument('--remote', action='store_true', help='Remote Execution')
    parser.add_argument('--s3', action='store_true', help='Retrieve Training Data from S3 instead')
    parser.add_argument('--data_proj_name', type=str, default='', help='dataset_project arg for ClearML Dataset')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    args = get_args() #Get Clearml args


    with open(args.train_config, 'r') as f:
        train_args = yaml.safe_load(f)

    Task.force_requirements_env_freeze(force=True, requirements_file='../requirements.txt')
    clearml_task = Task.init(project_name=PROJECT_NAME, task_name='syolov4_'+train_args['name'])
    if args.remote:

        # clearml_task.set_base_docker("nvcr.io/nvidia/pytorch:21.09-py3",
        #     docker_setup_bash_script=['pwd', 'ls', 'ls mish-cuda', 'cd mish-cuda', 'python setup.py build install', 'cd ..']
        # )
        # clearml_task.execute_remotely(queue_name="compute")

 
        clearml_task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04",
            docker_setup_bash_script=['pip3 show wheel', 'pip show wheel']
        )
        clearml_task.execute_remotely(queue_name="compute")

        cwd = os.getcwd()
        os.chdir('mish-cuda/dist')
        # subprocess.run('python3 -m pip install --upgrade pip'.split())
        subprocess.run(['pip3', 'install', 'mish_cuda-0.0.3-cp38-cp38-linux_x86_64.whl'])
        os.chdir(cwd)


        # # subprocess.run(["sh", "./install_mish.sh"])

        #  # Save current working directory, to jump back later after installation
        # # subprocess.run("git clone https://github.com/thomasbrandon/mish-cuda.git".split())
        # # os.chdir('mish-cuda')
        # # print(os.getcwd())
        # # print(os.listdir())
        # # subprocess.run(['python3', 'setup.py', 'build', 'install'])
        # # subprocess.call(['python3 setup.py build install', './install_mish.sh'])
        # # os.chdir(cwd)

        # pypath = os.environ['PYTHONPATH'].replace("::", ":")
        # os.environ['PYTHONPATH']=pypath+':/usr/local/lib/python3.8/dist-packages'
        # # subprocess.run(['export', 'PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3.8/dist-packages'])
        # for k, v in sorted(os.environ.items()):
        #     print(k+':', v)
        # print('\n')
        # subprocess.run(['pip', 'show', 'mish_cuda'])
        # subprocess.run(['pip', 'freeze'])
        import mish_cuda




    # only putting the import statement here because i need to install 
    # the damn mish before i could move on with life, otherwise in normal 
    # case where environments require no further setup, importing can be 
    # done entirely on the top
    import train_aip 
    
    ## Set Base Docker & bits for remote execution
    if args.s3:
        # Obtain Dataset 
        if args.data_proj_name=='':
            print("No data project name provided! Terminating...")
            exit()
        # If loading from S3, overwrite the train/val/test path with ClearML get_local_copy path
        with open(train_args['data'], 'r') as f:
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
        new_yaml_file = train_args['data'][:-5]+"_m.yaml" if train_args['data'][-5:]=='.yaml' else train_args['data'][:-4]+"_m.yml"
        with open(new_yaml_file, 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)
        train_args['data'] = new_yaml_file # use the modified yaml file

        # Get Pretrained Model
        pretrained_path = Dataset.get(
            dataset_name='pretrained-p5', 
            dataset_project='dataset/yolov4/models'
        ).get_local_copy()
        pretrained_path = os.path.join(pretrained_path, '')
        train_args['weights'] = pretrained_path+train_args['weights']


    args_list = []
    for x in train_args:
        # Handling Boolean Cases; ASSUMPTION: All boolean args are default store_true
        if type(train_args[x]) == bool and train_args[x] == False:
            continue
        if type(train_args[x]) == bool and train_args[x] == True:
            args_list.append('--'+x)
            continue
        args_list.append('--'+x)
        if type(train_args[x])==list:
            for k in train_args[x]:
                args_list.append(str(k))
        else: 
            args_list.append(str(train_args[x]))
    print(args_list)
    train_aip.main(args_list)

    ## To verify whether clearml is able to capture training through this means
    ## To return path to saved model from train_aip in local path for uploading, if s3=True
