#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站

from huggingface_hub import snapshot_download

model_name = "Yukang/Llama-2-7b-longlora-32k-ft"
model_path = os.path.join(os.getcwd(), model_name)
print(model_path)
# while True 是为了防止断联
while True:
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,  # 在local-dir指定的目录中都是一些“链接文件”
            cache_dir=model_path,
            token="",   # huggingface的token
            resume_download=True
        )
        break
    except:
        pass
