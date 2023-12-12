rm runs/detect
rm runs2
rm runs3

mkdir runs/detect
mkdir runs2
mkdir runs3

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python phase1.py --weights ./best.pt --img 480 --conf 0.4 --source ./sources --save-txt

python phase2.py

python phase3.py