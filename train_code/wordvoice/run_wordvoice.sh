#!/bin/bash

# Copyright 2024 Alibaba Inc. All Rights Reserved.
# 在运行本脚本前，请先下载 WordVoice-5A 数据集到 datasets/wordvoice-5a-zh 和 datasets/wordvoice-5a-en
# 若采用自己的数据集，请先采用 WordVoice-5A-Pipeline 处理至与 WordVoice-5A 数据集相同的格式

. ./path.sh || exit 1;
cd ../..

stage=0
stop_stage=3
CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"

lan=zh # en
data_dir=datasets/wordvoice-5a-$lan
wav_dir=datasets/wordvoice-5a-$lan

pdata_dir=datasets_processed/wordvoice-5a-$lan
pretrained_model_dir=checkpoints/Fun-CosyVoice3-0.5B

# dev test train-0 train-1 train-2 train-3 train-4
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
    for x in dev test train-0 train-1 train-2 train-3 train-4; do
        echo "🚀 处理 $x 数据"
        mkdir -p $pdata_dir/$x
        python xxh_tools/data_tools/prepare_data.py --src_dir $data_dir/$x.jsonl --des_dir $pdata_dir/$x --wav_dir $wav_dir
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Extract campplus speaker embedding, you will get utt2embedding.pt in $pdata_dir/$x dir"
    for x in dev test train-0 train-1 train-2 train-3 train-4; do
        echo "🚀 处理 $x 数据"
        python xxh_tools/data_tools/extract_embedding.py --dir $pdata_dir/$x \
            --onnx_path $pretrained_model_dir/campplus.onnx
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Extract discrete speech token, you will get utt2speech_token.pt in $pdata_dir/$x dir"
    for x in dev test train-0 train-1 train-2 train-3 train-4; do
        echo "🚀 处理 $x 数据"
        python xxh_tools/data_tools/extract_speech_token.py --dir $pdata_dir/$x \
            --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
    for x in dev test train-0 train-1 train-2 train-3 train-4; do
        echo "🚀 处理 $x 数据"
        mkdir -p $pdata_dir/$x/parquet
        python xxh_tools/data_tools/make_base_parquet.py --num_utts_per_parquet 800 \
            --num_processes 10 \
            --src_dir $pdata_dir/$x \
            --des_dir $pdata_dir/$x/parquet
    done
fi

# train llm
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Run train."
    export CUDA_VISIBLE_DEVICES="1"
    export TOKENIZERS_PARALLELISM=false
    # export CUDA_LAUNCH_BLOCKING=1

    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    job_id=1998
    dist_backend="nccl"
    num_workers=12
    prefetch=2
    train_engine=torch_ddp
    if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
    fi
    mkdir -p train_code/wordvoice/data

    cat ${pdata_dir}/{train-0,train-1,train-2,train-3,train-4}/parquet/data.list > train_code/wordvoice/data/train.data.list
    cp ${pdata_dir}/dev/parquet/data.list train_code/wordvoice/data/dev.data.list

    for model in llm; do # llm flow
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1266" \
        CosyVoice/cosyvoice/bin/train.py \
        --train_engine $train_engine \
        --config train_code/wordvoice/config/wordvoice.yaml \
        --train_data train_code/wordvoice/data/train.data.list \
        --cv_data train_code/wordvoice/data/dev.data.list \
        --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
        --onnx_path $pretrained_model_dir \
        --model $model \
        --checkpoint ${pretrained_model_dir}/${model}.pt \
        --model_dir train_results/exp/wordvoice_$model \
        --tensorboard_dir train_results/tensorboard/wordvoice_$model \
        --ddp.dist_backend $dist_backend \
        --num_workers ${num_workers} \
        --use_amp \
        --prefetch ${prefetch} \
        --pin_memory \
        --deepspeed_config train_code/wordvoice/config/ds_stage2.json \
        --deepspeed.save_states model+optimizer
    done
fi