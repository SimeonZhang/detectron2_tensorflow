#Detectron2 Tensorflow

[Detectron2](https://github.com/facebookresearch/detectron2) is FAIR's next generation software system that implements state-of-the-art object detection algorithms.
This repo reimplement most of the algorithms in Detectron2 with tensorflow and **reproduce** the expected results with the support of:

- Multi-GPU training
- Multi images per GPU
- Fast training and inference
- Flexibility to customize your own scripts to build the dataset or export the frozen model

In addition, some algorithms not included in detectron2 are also implemented, such as:

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575)
- [SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

## Dependencies

- Opencv, tensorflow >= 1.13.1
- pycocotools: `git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI` 
- [Pre-trained ResNet ImageNet model](https://pan.baidu.com/s/1rfkEi4jj81WDbHFeY_MCGw) (password: 6ia9) converted from detectron2

## Usage

Follow the [example config files](/configs) to make your own one. The config system is similar to detectron's.

- Prepare dataset:

  Scripts to prepare coco data are provided. Your own custom datasets should produce tensor dict following `TfExampleFields` in [fields](lib/data/fields.py).

  After giving the paths to raw and preprocessed data in a config file `example.yaml`, just run

  ```
  python build_tfrecords.py --config_file=example.yaml
  ```

- Train:

  ```bash
  python train.py --config_file=example.yaml
  ```

- Eval:

  ```bash
  python eval.py --config_file=example.yaml
  ```

- Export:

  ```bash
  python export.py --config_file=example.yaml
  ```

## Converted Model

Scripts to convert model from detectron2 (as well as some other projects) are also provided:

```bash
python convert_weights.py --config_file=example.yaml
```

The results of the converted models are quite similar.

You can download the [converted model](https://pan.baidu.com/s/1yAMl4m8UabjKa8lzOYgV6Q
) (password: mqbs) and fine-tune on your own datasets.

## Reference

* [detectron2](https://github.com/facebookresearch/detectron2)
* [object_detection_api](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [tensorpack](https://github.com/tensorpack/tensorpack)

