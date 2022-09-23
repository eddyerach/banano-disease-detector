# Banano leaf disease detector
## _Detectron2 based detector for aerial images_

This detector aims to detect chlorosis in banana leaves as an early symptom of disease.
## Features
- Detectron2 based
- Instance segmentation
- Augmented data as pixel-level transformation

## Installation
#### Docker

1. Pull the image from Docker Hub.

```sh
docker pull eddyerach1/detectron2_banano_uvas:latest
```

2. Build the container

```sh
docker run --gpus all -it -v /home:/host_home -w --name detectron2_banano detectron2_banano_uvas:latest
```
## Use
### Train
1. Configure paths and hiper parameters in the script [train.py]
2. Execute from container:
```sh
python train.py
```
### Test
1. Configure paths in the script [test.py]
```sh
python test.py
```

## License

MIT