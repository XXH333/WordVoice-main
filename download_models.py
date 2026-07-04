import torchaudio
from torchaudio.pipelines import MMS_FA
from huggingface_hub import snapshot_download

# 下载 MMS-FA 对齐模型
def download_mms_fa_model():
    print("正在下载 MMS-FA 对齐模型...")
    save_dir = "./checkpoints/mms_fa"
    bundle = MMS_FA

    model = bundle.get_model(
        with_star=False,
        dl_kwargs={"model_dir": save_dir}
    )

    print(f"模型 '{bundle}' 已成功下载到 '{save_dir}' 目录下。")

# 下载 CosyVoice3 模型
def download_cosyvoice3_model():
    print("正在下载 CosyVoice3 模型...")
    repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512" # 模型仓库ID
    local_path = './checkpoints/Fun-CosyVoice3-0.5B' # 本地下载路径
    snapshot_download(
        repo_id=repo_id, 
        local_dir=local_path,
        allow_patterns=["*"], # 指定下载的具体文件夹
        max_workers=2 # 限制并发
    )

    print(f"模型 '{repo_id}' 已成功下载到 '{local_path}' 目录下。")

def download_wordvoice_model():
    print("正在下载 WordVoice 模型...")
    repo_id="XXH333/WordVoice-base-0.5B" # 模型仓库ID
    local_path = './checkpoints/WordVoice-base-0.5B' # 本地下载路径
    snapshot_download(
        repo_id=repo_id, 
        local_dir=local_path,
        allow_patterns=["*"], # 指定下载的具体文件夹
        max_workers=2 # 限制并发
    )

    print(f"模型 '{repo_id}' 已成功下载到 '{local_path}' 目录下。")

def main():
    # download_mms_fa_model()
    # download_cosyvoice3_model()
    download_wordvoice_model()

if __name__ == "__main__":
    main()