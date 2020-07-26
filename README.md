# SSRNet
## Setup
Pre-prequisites for python project
```
python == 3.6
tensorflow >=1.3
joblib
trimesh
```
For C++ project

```
cudpp
lz4
flann_cuda
```
Build this version of Open3D(Rewritten by us)

```
$ cd Open3D
$ mkdir build
$ cd build
$ cmake ../src
$ make
```
Update the path to Open3D in python project

```
ssrnet/util/path_config.py
```
## Experiments
Experimental parameters are stored in .json configuration files.
We provide the prepared data for you. The transformed data consists of vertex labels, dividing information, and other necessary information. You can use it to run our network directly. 

You can put the data anywhere, but make sure the data path in your configuration file are set correctly.
## Training
To start network training, run

```
$ python ssr.py <config> --train
```
## Generate mesh
First, test a trained model to predict vertex labels. Run

```
$ python ssr.py <config> --test
```
Second, use the output label file to generate mesh. Run

```
$ mesh_generator <scan_path> <label_file> <output_mesh_file> 
```
Example:

```
$ mesh_generator .../scan001/ .../pre_label.txt .../output/ 
```
