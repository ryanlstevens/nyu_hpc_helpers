# NYU HPC Helpers

This is to keep track of tasks I've done to get things working on the NYU HPC. These tasks have been completed with the help of the NYU HPC staff (specifically Shenglong Wang), so much of the code comes from him. 

## Running Tensorflow in a Jupyter Notebook using Singularity and Slurm

To run Tensorflow in a Jupyter notebook, we have to accomplish the following steps:

1. Setup Conda environment inside singularity container
2. Install python packages 
3. Create Jupyter slurm run script 

### Setup Conda Environment 

We copy an (overlay) container image with a set amount of RAM and Storage space (.ext3), this will be the environment tensorflow will run in. We will then setup the conda environment inside the container, and finally a wrapper script we run when we reopen the container to point our environmental path to the conda environment.

A word of caution, we will be working with cuda version 10.1, this will require us to install an older version of tensorflow to get working.

```bash 

# Copy overlay container to current directory
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
gunzip overlay-5GB-200K.ext3.gz

# Setup conda environment inside container, running Ubuntu 18.04 with Cuda 10.1 
singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash

# Install miniconda environment 
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
export PATH=/ext3/miniconda3/bin:$PATH
conda update -n base conda -y

# Create wrapper script 
echo '''#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH''' > /ext3/env.sh
``` 

### Install Python packages 

We will now relaunch the container and install our python packages. Additionally, as a cleanup step, we rename our overlay container to __tensorflow.2.3.0.ext3__

```bash 
# Relaunch container 
singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
source /ext3/env.sh

# Now install python libraries 
pip install jupyter jupyterhub pandas matplotlib scipy 
pip install tensorflow==2.3.0

# Exit container and rename it 
mv overlay-5GB-200K.ext3 tensorflow.2.3.0.ext3
``` 

### Check that tensorflow install worked 

To check the installs worked, we run an interactive job with a gpu and then ask whether a gpu is available ``tf.test.is_gpu_available()``. If a GPU is available, this will return a true value.

```bash
# Run interactive slurm job 
srun -c2 --gres=gpu --mem='5gb' --pty /bin/bash

# Submit python command
singularity exec --nv --overlay tensorflow.2.3.0.ext3:ro \\ 
    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \\ 
    bash -c 'source /ext3/env.sh; python -c "import tensorflow as tf; print(tf.test.is_gpu_available());"'
```

### Get jupyter to 
