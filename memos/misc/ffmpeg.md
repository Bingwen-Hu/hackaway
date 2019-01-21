### extract audio from mp4:
```
ffmpeg -i 99.mp4 -f mp3 -vn 99.mp3
```

### merge several mp4 files to one

create a file contains thw following content 

```
file 'input1.mkv'
file 'input2.mkv'
```

then run
```
ffmpeg -f concat -i filelist.txt -c copy output.mkv
```


### extract key frame from video
```bash
ffmpeg -i video.mp4 -vf select='eq(pict_type\,I)' -vsync 2 -s 1920*1080 -f image2 core-%02d.jpeg
```


### extract all frame from video
```bash
ffmpeg -i video.mp4 pattern_%04d.jpg -hide_banner

# -hide_banner: hide ffmpeg compilation information
```

### extract a piece of video from video
```bash
ffmpeg  -i ./tocatoca.mp4 -vcodec copy -acodec copy -ss 00:00:10 -to 00:00:15 ./cutout1.mp4 -y
# -y means override if file exists
```


### combine image to a video
```bash
ffmpeg -f image2 -i face/%d.png -vcodec libx264 -r 30 -b 900k face.mp4
# -f format (image)
# -i input pattern
# -vcodec libx264 means mp4
# -r fps
# -b image size
```
