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