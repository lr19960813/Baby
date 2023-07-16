How to use this demo.


1. install pip tools

Only CLIP should be installed from the following command.
$ pip install git+https://github.com/openai/CLIP.git

If other tools are not installed, please install with pip.


2. preprocess

Uncomment preprocess.py of run.sh and run it.
If the file is already exist in the filelist directory, there is no need to execute.


3. training

Uncomment train.py of run.sh and run it.
Change the internals of config.py as needed.


4. evaluation

Uncomment eval.py of run.sh and run it.
When using umap for visualization, append the unfinished eval_umap.py and execute.
