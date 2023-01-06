$data_dir=../datasets/uaic2022_private_test/
rm -rf results
mkdir results
cd ABCnet
python tools/detect.py \
    --input ../datasets/uaic2022_private_test/images/ \
    --output ../results/ \
    --confidence_threshold 0.1 \
    --weights ../weights/model_0049999.pth
cd ..
python3 process_detection.py $data_dir
python3 predict_ocr.py $data_dir
python3 visualize_final.py