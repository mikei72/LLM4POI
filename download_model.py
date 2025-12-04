import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

def download_model(
    repo_id: str,
    local_dir: str = "./hf_models",
    token: str = "",
    max_retries: int = 10,
    retry_wait: int = 5
):
    save_path = os.path.join(local_dir, repo_id.replace("/", "_"))
    os.makedirs(save_path, exist_ok=True)

    print(f"\n=== 🚀 开始下载模型：{repo_id} ===")
    print(f"📂 保存路径：{save_path}")
    print(f"🔁 最大重试次数：{max_retries}\n")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"➡️  第 {attempt}/{max_retries} 次尝试下载……")

            snapshot_download(
                repo_id=repo_id,
                token=token if token else None,
                local_dir=save_path,
                local_dir_use_symlinks=False,   # 防止软链接，确保完整真实文件
                resume_download=True,           # 开启断点续传
            )

            print("\n🎉 下载成功！模型已保存至：", save_path)
            return save_path

        except Exception as e:
            print(f"⚠️ 错误：{e}")
            if attempt == max_retries:
                print("\n❌ 已达到最大重试次数，下载失败。")
                raise e
            print(f"⏳ {retry_wait} 秒后重试……\n")
            time.sleep(retry_wait)


download_model(
    repo_id="Yukang/Llama-2-7b-longlora-32k-ft",
    local_dir="./models"
)

