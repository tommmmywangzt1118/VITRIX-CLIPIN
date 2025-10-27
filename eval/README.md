# Evaluation
## Zero-shot classification/retrieval
The zero-shot classification and retrieval evaluation of VITRIX-CLIPIN are based on [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark), please make yourself ready for the environment and the evaluation dataset before you get started.
```python
git clone https://github.com/LAION-AI/CLIP_benchmark
pip install clip-benchmark
```
Then use the following to run the evaluation

```shell
MODEL_NAME=ViT-L-14-336-quickgelu
TASK=zeroshot_classification
DATASET=imagenet1k

cd vitl/clip_benchmark

python3 clip_benchmark/cli.py eval \
    --model $MODEL_NAME \
    --task $TASK \
    --pretrained /path/to/checkpoint \
    --dataset $DATASET \
    --output /path/to/output.json \
    --batch_size 32 \
    --language en \
    --dataset_root /path/to/dataroot \
    --long_cap
```
Please note that the `--long_cap` flag ensures the distilled rope features in our model.


