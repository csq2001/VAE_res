import os
from datasets import load_dataset
from PIL import Image

# =========================
# 1. 参数设置
# =========================
dataset_name = "conorcl/portraits-512"
save_dir = r"E:\Database_img\Protraits_real_512"
cache_dir = r"E:\HF_cache"

# 如果你想强制统一成 512×512，就保留下面这一行
# 如果想保持原图尺寸，就改成 None
target_size = (512, 512)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# =========================
# 2. 加载数据集
# =========================
print("Loading dataset...")
ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

print(ds)
print("Columns:", ds.column_names)
print("Total samples:", len(ds))
print("First sample keys:", ds[0].keys())

# =========================
# 3. 自动识别图片列
# =========================
image_col = None
sample = ds[0]

for col in ds.column_names:
    if isinstance(sample[col], Image.Image):
        image_col = col
        break

if image_col is None:
    raise RuntimeError("No image column found. Please inspect ds[0].")

print("Image column:", image_col)

# =========================
# 4. 批量保存图片
# =========================
saved = 0
failed = 0

for i, item in enumerate(ds):
    try:
        img = item[image_col].convert("RGB")

        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)

        out_path = os.path.join(save_dir, f"{i:06d}.png")
        img.save(out_path)
        saved += 1

    except Exception as e:
        failed += 1
        print(f"[Warning] failed on image {i}: {e}")

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(ds)} | saved={saved} | failed={failed}")

print("Finished.")
print("Saved:", saved)
print("Failed:", failed)
print("Output folder:", save_dir)