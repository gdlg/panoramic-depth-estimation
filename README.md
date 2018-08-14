# Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery
by Grégoire Payen de La Garanderie, Amir Atapour Abarghouei and Toby P. Breckon.

This is an implementation of the depth estimation approach presented in our paper on object detection and monocular depth estimation for 360° panoramic imagery. See our [project page](https://gdlg.github.io/panoramic) for more details. This implementation is an adaptation of the original implementation of [Monodepth](https://github.com/mrharicot/monodepth) of the paper “Unsupervised Monocular Depth Estimation with Left-Right Consistency” by Clément Godard et al. Please checkout their code and paper too.

## Citations
If you use our code, please cite our paper:

Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery
G. Payen de La Garanderie, A. Atapour Abarghouei, T.P. Breckon
*In Proc. European Conference on Computer Vision, Springer, 2018. [[arxiv]](https://arxiv.org/abs/1808.06253)*

## Installation

Install TensorFlow. We tested this code using TensorFlow 1.6.

## Inference

Our inference script is based on panoramas that have been cropped vertically due to memory contraints. To use the same crop as in our paper, take a 2048×1024 panorama and crop it vertically to the range [424,724] to get a 2048x300 panorama.

1. Download a dataset of 360° images such as [our dataset](https://hades.ext.dur.ac.uk/~greg/datasets/synthetic-panoramic-dataset.tar.gz) of synthetic images based on the CARLA simulator. You can also use images from Mapillary (see the [list](https://hochet.info/~gregoire/models/mapillary_image_keys.txt) of Mapillary image keys that we used for experimentation).

2. (optional). Fetch our pretrained models:
```
bash ./download_models.sh
```

3. Run the inference script `detection.py`. Here is an example using our pretrained model on the CARLA dataset from step 1.
```
mkdir my_output
python monodepth_simple.py --image_path ~/data/carla-dataset/test/image_2 --output_path my_output --checkpoint_path models/panoramic_checkpoints/mixed_warp/model-180000 --input_height 256 --input_width 1024
```

The input image is resized to the size specified by `--input_height` and `--input_width`. The resulting disparity maps are written to the directory specified by `--output_path`.

To convert the disparity to depth in meters, the inference script need to know the stereo baseline value. The default stereo baseline is the baseline that was used between the two colour cameras in the KITTI dataset. If you are using a different training dataset, you must specify the new stereo baseline in meters using the parameter `--stereo_baseline`. The stereo baseline must be specified regardless of whether the training dataset was based on rectilinear or equirectangular images.

## Training

To train using the regular code for rectilinear images:
```
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/kitti_raw/ --filenames_file utils/filenames/kitti_train_files.txt --log_directory mylog --rectilinear_mode
```

To train using the variant for crops of panoramic images as training input:
```
python monodepth_main.py --mode train --model_name my_model --data_path ~/data/kitti_raw/ --filenames_file utils/filenames/kitti_train_files.txt --log_directory mylog --fov 82.5
```

This second variant assumes that your input data is using an equirectangular projection (see `bilinear_sampler.py` for the details. To train on equirectangular images, unlike the rectilinear images, the angular resolution of the image must be known. This is done by specifying the horizontal field of view of the image using the `--fov` parameter.

Both of those training commands will output regular checkpoint in the specified checkpoint directory `mylog/my_model`.

Note: unlike the original monodepth code, we used PNG images as an input for training. If you would like to use JPEGs, you need to update the name of the image files in `utils/filenames/kitti_train_files.txt`. We used different data_path for each style-transfer variations of the KITTI datasets however we kept the same naming convention across all our variants and used the same filename_file for all of them.

## License

Our contributions are released under the MIT license. 
The original code is released under the [UCLB ACP-A License](https://github.com/mrharicot/monodepth). Please see the LICENSE file for more details.

