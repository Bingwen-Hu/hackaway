# refer [[this]](https://medium.com/@vitali.usau/install-cuda-10-0-cudnn-7-3-and-build-tensorflow-gpu-from-source-on-ubuntu-18-04-3daf720b83fe) and [[this]](https://blog.csdn.net/EliminatedAcmer/article/details/80528980)


## install CUDA and Driver
1. **get to its site and download cuda.run file and cudnn.tar file**
2. **remove nouveau**

    run 

    ```
    lsmod | grep nouveau
    ``` 

    if there is output, run
    ```
    cd /etc/modprobe.d
    sudo touch blacklist-nouveau.conf
    sudo gedit blacklist-nouveau.conf
    ```
    in the opened file, type in
    ```
    blacklist nouveau  
    options nouveau modeset=0 
    ```
    save and exit, then run on terminal
    ```
    sudo update-initramfs -u
    ```
    reboot and check with
    ```
    lsmod | grep nouveau
    ```
    if there is no output, we can step ahead.

3. **install driver**

    first, uninstall elder driver
    ```
    sudo apt-get remove --purge nvidia-*
    ```
    then C+M+F! into text 
    ```
    sudo service lightdm stop
    ```
    install driver
    ```
    sudo chmod a+x NVIDIA-Linux-x86_64-396.18.run
    sudo ./NVIDIA-Linux-x86_64-396.18.run –no-x-check –no-nouveau-check–no-opengl-files
    ```
    reboot and check with
    ```
    nvidia-smi
    ```
4. **install cuda**

    cd into to directory where runfile stayed
    ```
    sudo sh cuda_9.2.88_396.26_linux.run
    ```
    it will ask you whether install driver, choose no. Finally, edit the path
    ```
    sudo gedit  /etc/profile
    ```
    type in
    ```
    export  PATH=/usr/local/cuda-9.2/bin:$PATH
    export  LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH　
    ```
    reboot and test it
    ```
    cd  /usr/local/cuda-9.2/samples/1_Utilities/deviceQuery
    sudo make
    ./deviceQuery
    ```
5. **install cudnn**

    this may work.
    ```
    sudo dpkg -i libcudnn7_7.3.0.29–1+cuda10.0_amd64.deb
    ```
    if it doesn't work, unpackage taz file and mv cudnn.h into cuda/include and cudnnn* into cuda/lib64, symbol link should be reset. Following cmd may help while compile tensorflow.
    ```
    sudo ldconfig /usr/local/cuda-10.0/lib64
    ```

## Install Bazel
install tool curl
```
sudo apt install curl
```
and install Bazel using
```
sudo apt-get install openjdk-8-jdk
echo “deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8” | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install bazel
```



## Build TensorFlow

1. **Activate virtual environment and run following commands:**
    ```
    pip install -U pip six numpy wheel mock
    pip install -U keras_applications==1.0.5 --no-deps
    pip install -U keras_preprocessing==1.0.3 --no-deps
    ```
2. **install git and clone tensorflow**
    ```
    sudo apt install git
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ```
    checkout target version
    ```
    cd tensorflow
    git checkout r1.12
    ```

3. **Configure TensorFlow build**

    in the root directory, running 
    ```
    ./configure
    ```
    choose using CUDA, other is ... not very important...

4. **Build and install TensorFlow.** 
    ``` 
    sudo ldconfig /usr/local/cuda-10.0/lib64
    cat tensorflow/tools/bazel.rc >> tensorflow/.tf_configure.bazelrc
    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    ```
    After building successfully
    ```
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    ```
    after hours and package is ready, rung
    ```
    pip install tmp/tensorflow_pkg/tensorflow-version-*-linux_x86_64.whl
    ```


