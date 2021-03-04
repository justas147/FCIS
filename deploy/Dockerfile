FROM nvidia/cuda:8.0-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install build-essential -y && \
    apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 libglib2.0-0 -y && \
    apt install wget git unzip -y && \
    apt-get libopencv-dev -y && \
	# apt-get install libopenblas-dev pkg-config -y && \
	cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    bash ~/Miniconda2-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda2-latest-Linux-x86_64.sh && \
    ~/miniconda2/bin/conda init && \ 
    exec bash && \
    pip install Cython && \
    pip install opencv-python==3.2.0.6 && \
    pip install easydict==1.6 && \
    pip install hickle && \
    pip install pyyaml && \
    pip install Pillow && \
	pip install matplotlib && \
    cd ~ && \
    git clone https://github.com/msracver/FCIS.git
	
	# cd ~/miniconda2/lib/python2.7/site-packages && \
	# git clone --recursive https://github.com/dmlc/mxnet.git && \
	# cd mxnet/ && \
	# git checkout 998378a && \
	# git submodule init && \
	# git submodule update && \
	# cp -r ~/FCIS/fcis/operator_cxx/channel_operator* ~/miniconda2/lib/python2.7/site-packages/mxnet/src/operator/contrib/ && \
	# make -j 4 USE_OPENCV=1 USE_BLAS=openblas && \
	# cd ~/miniconda2/lib/python2.7/site-packages/mxnet/python/ && \
	# python setup.py install

ADD mxnet ~/miniconda2/lib/python2.7/site-packages/mxnet

RUN cd ~/miniconda2/lib/python2.7/site-packages/mxnet/python/ && \
    python setup.py install

COPY coco_model.zip ~/FCIS/model

RUN cd ~/FCIS/model && \
	sh ./init.sh && \
    unzip coco_model.zip && \
    rm coco_model.zip && \
    cd coco_model && \
    mv fcis_coco-0000.params .. && \
    cd .. && \
    rm -d coco_model && \