#!/usr/bin/env python3

import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch


def job(utt_list, parquet_file, utt2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read()
        data_list.append(data)

    # 保存到parquet,utt2parquet_file
    df = pd.DataFrame()
    df['utt'] = utt_list
    df['audio_data'] = data_list
    df['wav'] = [utt2wav[utt] for utt in utt_list]
    df['text'] = [utt2text[utt] for utt in utt_list]
    df['words'] = [[word['word'] for word in utt2words[utt]] for utt in utt_list]
    df['start'] = [[word['start'] for word in utt2words[utt]] for utt in utt_list]
    df['end'] = [[word['end'] for word in utt2words[utt]] for utt in utt_list]
    df['f0'] = [utt2f0[utt] for utt in utt_list]
    df['energy'] = [utt2energy[utt] for utt in utt_list]
    df['tone'] = [utt2tone[utt] for utt in utt_list]
    df['boundary'] = [utt2boundary[utt] for utt in utt_list]

    if utt2embedding is not None:
        df['utt_embedding'] = [utt2embedding[utt] for utt in utt_list]
    if utt2speech_token is not None:
        df['speech_token'] = [utt2speech_token[utt] for utt in utt_list]
    if utt2instruct is not None:
        df['instruct'] = [utt2instruct[utt] for utt in utt_list]
    if args.dpo:
        df['reject_speech_token'] = [utt2reject_speech_token.get(utt, None) for utt in utt_list]
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    args = parser.parse_args()

    with open('{}/wav.json'.format(args.src_dir)) as f:
        utt2wav = json.load(f)
    with open('{}/text.json'.format(args.src_dir)) as f:
        utt2text = json.load(f)
    with open('{}/utt2words.json'.format(args.src_dir)) as f:
        utt2words = json.load(f)
    with open('{}/utt2f0.json'.format(args.src_dir)) as f:
        utt2f0 = json.load(f)
    with open('{}/utt2energy.json'.format(args.src_dir)) as f:
        utt2energy = json.load(f)
    with open('{}/utt2tone.json'.format(args.src_dir)) as f:
        utt2tone = json.load(f)
    with open('{}/utt2boundary.json'.format(args.src_dir)) as f:
        utt2boundary = json.load(f)

    if os.path.exists('{}/instruct'.format(args.src_dir)):
        utt2instruct = {}
        with open('{}/instruct'.format(args.src_dir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt2instruct[l[0]] = ' '.join(l[1:])
    else:
        utt2instruct = None
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/utt2embedding.pt'.format(args.src_dir)) else None
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}/utt2speech_token.pt'.format(args.src_dir)) else None
    if args.dpo:
        utt2reject_speech_token = torch.load('{}_reject/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}_reject/utt2speech_token.pt'.format(args.src_dir)) else {}
    utts = list(utt2wav.keys())

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list= [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file))
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
