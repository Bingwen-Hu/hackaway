# today deserve a cheer

## how I install caffe and its python interface


### prerequires
install as the docs

### modifications
+ Makefile.config
  + uncomment opencv: = 3
  + uncomment python3
  + remove the first two line in CUDA_ARCH since my cuda > 9.0 
  + enable python layer
  + update boost_python.so to link to boost_python3.so
  + add /usr/include/hdf5/serial/ to INCLUDE_DIRS

+ Makefile
  + rename *hdf5_hl hdf5* to hdf5_serial_hl hdf5_serial
  + add *opencv_imgcodecs* to opencv section

+ others 
  + compile opencv and caffe with the same version gcc(6.4)
  + install libtcmalloc_minimal and setup LD_PRELOAD
