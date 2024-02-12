GPU programming
===============

A tool for interactive GPU learning. It provides a guide and jupyter notebooks based on https://github.com/DeiC-HPC/gpu-jupyter.


## Building image
The singularity image can be build using Nix flakes with the command `nix build .#singularity-nginx`, if you want to give the image a certain name then add `-o {name}`.

The image is then run as any other singularity image.