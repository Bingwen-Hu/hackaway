### nontrival introduction 
demo design using [face_recognition](https://github.com/ageitgey/face_recognition)


### TODOLIST
+ [x] build the face database
+ [x] using yolo detect the bbox
+ [ ] postprocess to make the matching people show and other hide
+ [ ] output the mp4 -- that's all

### dataflow
image --> yoloapi --> select one person bbox --> faceapi --> recognize person info 