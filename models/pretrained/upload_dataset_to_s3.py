from clearml import Dataset


dataset = Dataset.create(dataset_name='pretrained-p5', dataset_project = 'datasets/yolov4/models')
dataset.add_files('yolov4-p5_.pt')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()
dataset.publish()

dataset = Dataset.create(dataset_name='pretrained-p6', dataset_project = 'datasets/yolov4/models')
dataset.add_files('yolov4-p6_.pt')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()
dataset.publish()


dataset = Dataset.create(dataset_name='pretrained-p7', dataset_project = 'datasets/yolov4/models')
dataset.add_files('yolov4-p7.pt')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()
dataset.publish()