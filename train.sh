pwd
cd ./train
git clone https://github.com/ultralytics/yolov5
pip install -q roboflow
pip install roboflow
pip install -qr requirements.txt
cp phase1.py ./yolov5
cd yolov5

python ./train1.py

python ./train3.py