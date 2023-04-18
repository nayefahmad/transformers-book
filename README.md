# Transformers Playground

This repo is for exploring `transformers` models, largely based on the [book by Lewis Tunstall, Leandro von Werra, and Thomas Wolf](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).  

## Notes 

### Using Docker for HF libraries 

I couldn't install hugginface libraries on my laptop directly, so I resorted to running code inside a docker container. This is the [Docker image with hf libraries](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu). 

Steps: 

1. In PowerShell, make sure you have Docker installed. Run `docker --version`
1. Run `docker pull huggingface/transformers-pytorch-gpu`
1. (Optional) Just to explore a container from this image, you can run `docker run -it huggingface/transformers-pytorch-gpu /bin/bash`. For example, you can get a python interpreter by running `python3`. The default interpreter has the hf libraries installed, so you can immediately run e.g. `import transformers`. 
1. In PyCharm, add python interpreter > Docker. Image name is `huggingface/transformers-pytorch-gpu`. Python interpreter path is `/bin/python3.8`. This interpreter already has the hf libraries installed, so no need to pip install them. Note: you can't install any new python libraries in this interpreter, so use it only for hf functionality. If you need other libraries, switch interpreters. 
