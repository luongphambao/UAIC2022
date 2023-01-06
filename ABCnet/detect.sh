python tools/detect.py \
    --input /app/UAIC/datasets/uaic2022_private_test/images/ \
    --output /app/UAIC/results/ \
    --confidence_threshold 0.1 \
    --weights /app/UAIC/ABCnetV2/model_v2_totaltext.pth


python tools/detect.py \
    --input /app/UAIC/datasets/uaic2022_private_test/images/ \
    --output /app/UAIC/results/ \
    --confidence_threshold 0.3 \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --weights /app/UAIC/ABCnetV2/model_final.pth
# python demo/demo.py \
#     --config-file /app/UAIC/AdelaiDet/configs/BAText/CustomText/attn_R_50.yaml \
#     --input /app/UAIC/datasets/uaic2022_private_test/images/ \
#     --output /app/UAIC/ \
#     --confidence-threshold 0.3 \
#     --opts MODEL.WEIGHTS /app/UAIC/model_final.pth