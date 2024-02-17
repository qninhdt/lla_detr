export TORCH_COMPILE_DEBUG=1

python src/eval.py \
	model=wa_detr \
    data=bdd100k \
    data.limit=0.1 \
    data.batch_size=2 \
    data.num_workers=8 \
    data.image_size=[540,960] \
    trainer=gpu \
    +trainer.precision="16-mixed" \
    model.compile=false \
    ckpt_path="./checkpoints/lla_detr_0.ckpt"