#train
python tools/train.py configs/tps_resnet_bilstm_attn.py "0"
python tools/train.py configs/small_satrn.py "0"
#inference
# inference using GPUs with gpu_id 0
python tools/inference.py configs/tps_resnet_bilstm_attn.py weights/tps_resnet_bilstm_attn/best_norm.pth test/ "0"
