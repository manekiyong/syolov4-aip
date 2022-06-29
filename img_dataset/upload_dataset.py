from clearml import Dataset

dataset_proj_name = 'datasets/yolov4/<INSERT NAME HERE>'

dataset = Dataset.create(dataset_name='train', dataset_project = dataset_proj_name)
dataset.add_files('./train/')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()

dataset = Dataset.create(dataset_name='val', dataset_project = dataset_proj_name)
dataset.add_files('./val/')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()

dataset = Dataset.create(dataset_name='test', dataset_project = dataset_proj_name)
dataset.add_files('./test/')
dataset.upload(output_url='s3://experiment-logging/')
dataset.finalize()