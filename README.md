# RaCAV-eval

This repository contain the Retrieval-augmented Code Generability (RaCG) evaluation framework. Specifically, this repo includes benchmak dataset, pre-trained verification models checkpoints, inference and evaluation script.

## RaCAV dataset
RaCAV dataset is designed to evaluate answerability verification task for code generation task with RAG scenario. See the [paper]() for more details.

### Example
```
"RaCAV/697": {
    "query": "Can you guide me through compressing my custom model, retraining it, and measuring its latency?",
    "retrieved_APIs": {
        "API_1": "Compressor.automatic_compression(input_model_path: str, output_dir: str, input_shapes: List[Dict[str, int]], framework: Framework = Framework.PYTORCH, compression_ratio: float = 0.5): Compress a model automatically based on the given compression ratio. As the compression ratio increases, you can get more lighter and faster compressed models, but with greater lost of accuracy. Therefore, it is necessary to find an appropriate ratio for your requirements. It might require a few trials and errors. The range of available values is as follows(0 < ratio <=1). Returns source model and compressed model information.\n\n\nParameters\n----------\ninput_model_path (str): The file path where the model is located.\noutput_dir (str): The local path to save the compressed model.\ninput_shapes (List[Dict[str, int]]): Input shapes of the model.\nframework (Framework, optional): The framework of the model.\ncompression_ratio (float, optional): The compression ratio for automatic compression. Defaults to 0.5.\n\n\nExamples\n----------\nfrom netspresso import NetsPresso\n\n\nnetspresso = NetsPresso(email='YOUR_EMAIL', password='YOUR_PASSWORD')\n\ncompressor = netspresso.compressor()\ncompressed_model = compressor.automatic_compression(\n    input_shapes=[{'batch': 1, 'channel': 3, 'dimension': [224, 224]}],\n    input_model_path='./examples/sample_models/graphmodule.pt',\n    output_dir='./outputs/compressed/pytorch_automatic_compression_1',\n    compression_ratio=0.5,\n)\n\nParameter Candidates\n--------\n```python\nclass Framework(str, Enum):\n    TENSORFLOW_KERAS = 'tensorflow_keras'\n    TENSORFLOW = 'saved_model'\n    PYTORCH = 'pytorch'\n    ONNX = 'onnx'\n    TENSORRT = 'tensorrt'\n    OPENVINO = 'openvino'\n    TENSORFLOW_LITE = 'tensorflow_lite' # TFLite\n    DRPAI = 'drpai'\n```\n",
        "API_2": "Trainer.train(gpus: str, project_name: str): Train the model with the specified configuration. Returns a dictionary containing information about the training.\n\n\nParameters\n----------\ngpus (str): GPU ids to use, separated by commas.\nproject_name (str): Project name to save the experiment.\n\n\nExamples\n----------\nfrom netspresso import NetsPresso\nfrom netspresso.enums import Task\nfrom netspresso.trainer.augmentations import Resize\nfrom netspresso.trainer.optimizers import AdamW\nfrom netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp\n\n\nnetspresso = NetsPresso(email='YOUR_EMAIL', password='YOUR_PASSWORD')\n\ntrainer = netspresso.trainer(task=Task.OBJECT_DETECTION)\n\ntrainer.set_dataset_config(\n    name='traffic_sign_config_example',\n    root_path='/root/traffic-sign',\n    train_image='images/train',\n    train_label='labels/train',\n    valid_image='images/valid',\n    valid_label='labels/valid',\n    id_mapping=['prohibitory', 'danger', 'mandatory', 'other'],)\n\n\ntrainer.set_model_config(model_name='YOLOX-S', img_size=512)\n\ntrainer.set_augmentation_config(\n    train_transforms=[Resize()],\n    inference_transforms=[Resize()],\n)\n\noptimizer = AdamW(lr=6e-3)\nscheduler = CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10)\ntrainer.set_training_config(\n    epochs=40,\n    batch_size=16,\n    optimizer=optimizer,\n    scheduler=scheduler,)\n\ntraining_result = trainer.train(gpus='0, 1', project_name='project_sample')\n\nParameter Candidates\n--------\n```python\nclass Task(str, Enum):\n    IMAGE_CLASSIFICATION = 'classification'\n    OBJECT_DETECTION = 'detection'\n    SEMANTIC_SEGMENTATION = 'segmentation'\n```\n\navailable model names: EfficientFormer, YOLOX-S, ResNet, MobileNetV3, MixNetL, MixNetM, MixNetS\n\noptimizers: Adadelta, Adagrad, Adam, Adamax, AdamW, RMSprop, SGD\n\n\nschedulers: StepLR, PolynomialLRWithWarmUp, CosineAnnealingLRWithCustomWarmUp, CosineAnnealingWarmRestartsWithCustomWarmUp\n\n\naugmentations: ColorJitter, Pad, RandomCrop, RandomCutmix, RandomHorizontalFlip, RandomMixup, RandomResizedCrop, RandomVerticalFlip, Resize, TrivialAugmentWide\n",
        "API_3": "Benchmarker.benchmark_model(input_model_path: str, target_device_name: DeviceName, target_data_type: DataType = DataType.FP16, target_software_version: Optional[Union[str, SoftwareVersion]] = None, target_hardware_type: Optional[Union[str, HardwareType]] = None, wait_until_done: bool = True): Benchmark the specified model on the specified device. Returns model benchmark task dictionary. Please refer to this link(https://github.com/Nota-NetsPresso/PyNetsPresso/blob/develop/SUPPORT_OPTIONS.md) to check device/software/framework compatibility.\n\n\nParameters\n----------\ninput_model_path (str): The file path where the model is located.\ntarget_device_name (DeviceName): Target device name.\ntarget_data_type (DataType): Data type of the model.\ntarget_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.\ntarget_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.\nwait_until_done (bool): If True, wait for the conversion result before returning the function. If False, request the conversion and return the function immediately.\n\n\nDetails of return\n--------\nstatus: Indicates the current status of the task. task_type: Describes the type of task being performed. \nbenchmark_info: A nested object containing detailed information about the benchmarking task \ntarget_device: Specifies the device used for the benchmark. \nfilename: The name of the file being used in the benchmark. \ndata_type: The data type of the model used in the benchmark. \nprocessor: Indicates the processor architecture. \nsoftware_version: The version of the software of the device used. \nhardware_type: The type of hardware used. \ninput_model_uuid: A unique identifier for the input model used in the benchmark. \nbenchmark_task_uuid: A unique identifier for the benchmark task itself. \ndevicefarm_benchmark_task_uuid: A unique identifier for the benchmark task that runs on a device farm. \ndevicefarm_model_uuid: A unique identifier for the used model on a device farm. \nresult: A nested object containing the results from the benchmark. \nmemory_footprint_gpu: Memory usage on the GPU during the benchmark, measured in MB. \nmemory_footprint_cpu: Memory usage on the CPU during the benchmark, measured in MB. \nlatency: The time taken to complete the benchmark, measured in ms. \nram_size: The RAM size occupied on the device during the benchmark, measured in MB. \npower_consumption: Power consumed during the benchmark, measured in kW. file_size: The size of the file used in the benchmark, measured in MB.\n\n\nExamples\n----------\nfrom netspresso import NetsPresso\nfrom netspresso.enums import DeviceName, SoftwareVersion\n\n\nnetspresso = NetsPresso(email='YOUR_EMAIL', password='YOUR_PASSWORD')\nbenchmarker = netspresso.benchmarker()\nbenchmark_task = benchmarker.benchmark_model(\n    input_model_path='./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1.trt',\n    target_device_name=DeviceName.JETSON_AGX_ORIN,\n    target_software_version=SoftwareVersion.JETPACK_5_0_1,\n)\n\nParameter Candidates\n--------\n```python\nclass DeviceName(str, Enum):\n    RASPBERRY_PI_5 = 'RaspberryPi5'\n    RASPBERRY_PI_4B = 'RaspberryPi4B'\n    RASPBERRY_PI_3B_PLUS = 'RaspberryPi3BPlus'\n    RASPBERRY_PI_ZERO_W = 'RaspberryPi-ZeroW'\n    RASPBERRY_PI_ZERO_2W = 'RaspberryPi-Zero2W'\n    RENESAS_RZ_V2L = 'rzv2l_avnet'\n    RENESAS_RZ_V2M = 'rzv2m'\n    JETSON_NANO = 'Jetson-Nano'\n    JETSON_TX2 = 'Jetson-Tx2'\n    JETSON_XAVIER = 'Jetson-Xavier'\n    JETSON_NX = 'Jetson-Nx'\n    JETSON_AGX_ORIN = 'Jetson-AGX-Orin'\n    AWS_T4 = 'AWS-T4'\n    INTEL_XEON_W_2233 = 'Intel-Xeon'\n    ALIF_ENSEMBLE_E7_DEVKIT_GEN2 = 'Ensemble-E7-DevKit-Gen2'\n    RENESAS_RA8D1 = 'Renesas-RA8D1'\n    ARM_ETHOS_U_SERIES = 'Arm Virtual Hardware Ethos-U Series'\n```\n\n```python\nclass SoftwareVersion(str, Enum):\n    JETPACK_4_4_1 = '4.4.1-b50'\n    JETPACK_4_6 = '4.6-b199'\n    JETPACK_5_0_1 = '5.0.1-b118'\n    JETPACK_5_0_2 = '5.0.2-b231'\n```\n\n```python\nclass HardwareType(str, Enum):\n    HELIUM = 'helium'\n```\n\n```python\nclass DataType(str, Enum):\n    FP32 = 'FP32'\n    FP16 = 'FP16'\n    INT8 = 'INT8'\n    NONE = ''\n```\n"
    },
    "label": "Answerable",
    "library": "NetsPresso",
    "canonical_solution": "from netspresso import NetsPresso \nfrom netspresso.trainer.optimizers import AdamW \nfrom netspresso.enums import DeviceName, SoftwareVersion \nnetspresso = NetsPresso(email='YOUR_EMAIL', password='YOUR_PASSWORD') \ncompressor = netspresso.compressor() \ncompressed_model = compressor.automatic_compression(input_shapes=[{'batch': 1, 'channel': 3, 'dimension': [224, 224]}], input_model_path='./models/YOUR_PYTORCH_MODEL.pt', output_dir='/outputs/compressed', compression_ratio=0.5) \ntrainer = netspresso.trainer(yaml_path='hparams.yaml') \ntrainer.set_fx_model(fx_model_path='YOUR_FX_MODEL.pt') \ntrainer.set_training_config(epochs=100, batch_size=128, optimizer=AdamW(lr=1e-4)) \ntrainer.train(gpus='0,1', project_name='project_retrain') \nbenchmarker = netspresso.benchmarker() \nbenchmark_result = benchmarker.benchmark_model(input_model_path='YOUR_RETRAINED_MODEL_PATH', target_device_name=DeviceName.JETSON_AGX_ORIN, target_software_version=SoftwareVersion.JETPACK_5_0_1)",
},
```

### Annotation Criteria
We define annotation criteria for determining whether a user query can be answered based on a specified API database. User queries usually consist of multiple requests. If all of the requests can be resolved using the library's APIs, the query is annotated as **answerable**. If only some of the requests can be resolved, it is annotated as **partially answerable**. If none of the requests can be resolved, it is annotated as **unanswerable**.

### Selected Libraries
We selected *private libraries* that code language models have not been trained on. These *private libraries* ensure that the evaluation of answerability verification is independent of the prior knowledge of the code language models. Since code language models lack prior knowledge of the private libraries, they should heavily rely on retrieved APIs to generate responses for user queries. RaCAV includes the following 4 libraries:

- [**NetsPressoEval**](https://nota-netspresso.github.io/PyNetsPresso/description.html): Library for training, compressing, deploying, and benchmarking neural models on various hardware.
- [**TorchDataEval**](https://github.com/microsoft/PyCodeGPT/tree/main/apicoder): Originated from TorchData, a beta library for modular data loading framework and efficient data pipelines.
- [**BeatNumEval**](https://github.com/microsoft/PyCodeGPT/tree/main/apicoder): Library based on Python Numpy but with keywords and structure manually paraphrased.
- [**MonkeyEval**](https://github.com/microsoft/PyCodeGPT/tree/main/apicoder): Library based on Python Pandas but with keywords and structure manually paraphrased.

Each library consists of a user's query, gold APIs, canonical solution, and test code. They are designed for evaluating LLM's code generation quality given the user's request.

### Building unanswerable/partially answerable samples
Unanswerable and partially answerable samples correspond to cases where all or some of the requests in the user query cannot be resolved using the retrieved APIs. Therefore, we convert the answerable samples to unanswerable and partially answerable samples using the following operations:

- Substitute gold APIs
- Merge queries with related topics
- Query from out-of-database

### Dataset statistics
The number of samples in the RaCAV dataset for each library and answerability type is shown in the table below. Every generated query is cross-checked by the 4 annotators.

| Libraries | #Answerable | #Partially answerable | #Unanswerable | Canonical solution | Test code |
|:---:|:---:|:---:|:---:|:---:|:---:|
| NetsPressoEval | 70 | 49 | 121 | O | X |
| TorchDataEval | 50 | 44 | 83 | O | O |
| BeatNumEval | 85 | 89 | 148 | O | O |
| MonkeyEval | 78 | 59 | 140 | O | O |


## Installation
```bash
conda create -yn racav-env python=3.11
conda activate racav-env
pip install -r requirements.txt
```

## Pre-trained models
Pre-trained checkpoints (as an form of adapter) are available from [llama3-8b-adapter-RaCAV](https://huggingface.co/nota-ai/llama3-8b-adapter-RaCAV), and [gemma-7b-adapter-RaCAV](https://huggingface.co/nota-ai/gemma-7b-adapter-RaCAV).

## Inference
Access to [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Gemma](https://huggingface.co/google/gemma-1.1-2b-it) are required.
You can get a response using `infer.py` with the example input in the code.

```bash
python infer.py --model_name {gemma-7b, llama3-8b} --token {HF_token}
```

## Benchmark

|            | Zero-shot | + Fine-tuning (QLoRA) | + In-context learning (3way-1shot) |
|------------|-------------------|--------------------|---------------------|
| Llama 3 8B | 33.0             | 36.5             | 53.7               |
| Gemma 7B   | 36.9             | 46.7             | 55.8               |

## Terms of use
The dataset published in this repository contains several original datasets ([license](https://github.com/microsoft/PyCodeGPT/blob/main/apicoder/APIRetriever/LICENSE)) including TorchDataEval, BeatNumEval, andMonkeyEval) and NetsPressoEval. Commercial use of any modifications or additions made from the original datasets is not allowed. 
