### pull image
```bash
docker pull blvc/caffe
```

### run in interactive mode
```bash
docker run -it blvc/caffe bash
```


### list image
docker image ls


### delete local image
```bash
docker image rm xxx
docker image rm $(docker image ls -q redis)
```