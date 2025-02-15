cd ..
del .\distributed\environment

python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --config data_configs/sequences.yaml --device cpu --dist-url "file:///distributed/environment" --world-size 2 --workers 2 --epochs 100 --model fasterrcnn_resnet50_fpn --no-mosaic -imw 10000 -imh 2249 --project-name scalograms --batch-size 2 -dw
