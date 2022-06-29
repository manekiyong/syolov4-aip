# Fine-tuning of Scaled YOLOv4 

### Environment Setup
1) Create and activate Python 3.8 venv
2) Run `pip install -r requirements.txt`
3) Prepare dataset specific `.yaml` file (Refer to data folder for example)
    * If dataset is on S3, a placeholder text can be placed in the `.yaml` file.
    * The dataset location will be indicated as arguments when executing `main.py`.
4) Download pretrained Scaled YOLOv4 model from [here](https://github.com/WongKinYiu/ScaledYOLOv4).
    * Pretrained models are available on ClearML too. The default model used is p5 model. Refer to clearml on the other pretrained models uploaded, and change it accordingly on Line 72 of `main.py`

### Dataset Preparation (img_dataset folder)
Refer to img_dataset folder for dataset preparation. 
* For each image in train, val and test, it must be accompanied by a text file with the class, and bounding boxes coordinates for each object within the image. Example `.txt` file is shown in the folders. 
* The name of the text file must correspond to the image file. (i.e. `00a9f1817594f581.jpg` -> `00a9f1817594f581.txt`)
* Upload the dataset to S3 using the `upload_dataset.py` script. 

### Execution (src folder)
Refer to datasets/yolov4/ on ClearML to see pre-uploaded dataset available (googleoi, brainhack)

* Local Execution (Requires torch and torchvision in venv, if not installed):
    * `python train_aip.py --weights yolov4-p5_.pt --data data/<dataset>.yaml --device 0 --epochs 1 --batch 8 --nosave --name <exp name>`
* Local Execution & Retrieve files from S3:
    * `python main.py --train_config config.yaml --data_proj_name datasets/yolov4/<dataset> --s3`
* Remote Execution & Retrieve files from S3:
    * `python main.py --train_config config.yaml --data_proj_name datasets/yolov4/<dataset> --s3 --remote`