import os
from PIL import Image

# 基本路徑
base_path = "C:/Users/ouche/Desktop/CV_FP/train"

# 定義擴增方式
augmentations = {
    "_left5": lambda img: img.rotate(5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_left10": lambda img: img.rotate(10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right5": lambda img: img.rotate(-5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right10": lambda img: img.rotate(-10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_mirror": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "_left5m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_left10m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right5m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-5, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_right10m": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-10, fillcolor=(0 if img.mode == "L" else (255, 255, 255))),
    "_zoom_in": lambda img: zoom_in(img, scale=1.2),  # 方法1：放大並切割
    "_zoom_out": lambda img: zoom_out(img, scale=0.8)  # 方法2：縮小並填充黑色
}

# 放大並裁剪圖片
def zoom_in(img, scale=1.2):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 放大圖像
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 計算裁剪範圍，保持原圖大小
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    return img_resized.crop((left, top, right, bottom))

# 縮小並填充黑色區塊
def zoom_out(img, scale=0.8):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 縮小圖像
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 創建黑色背景圖片
    new_img = Image.new("RGB", (width, height), (0, 0, 0))

    # 計算縮小後圖像放置的位置
    left = (width - new_width) // 2
    top = (height - new_height) // 2

    # 將縮小後的圖像粘貼到黑色背景上
    new_img.paste(img_resized, (left, top))

    return new_img

# 遍歷資料夾
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        print(f"{folder_path} 不是資料夾，跳過。")
        continue

    # 處理每張圖片
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"無法讀取圖片 {img_path}，錯誤：{e}")
            continue

        # 執行每種擴增操作並將圖片儲存回原始資料夾
        for aug_suffix, aug_function in augmentations.items():
            aug_img = aug_function(img)  # 擴增圖片

            # 裁剪至原始尺寸
            original_width, original_height = img.size
            new_width, new_height = aug_img.size

            left = (new_width - original_width) // 2
            top = (new_height - original_height) // 2
            right = left + original_width
            bottom = top + original_height

            aug_img = aug_img.crop((left, top, right, bottom))

            # 儲存圖片到原始資料夾中，名稱不變
            aug_img_path = os.path.join(folder_path, f"{os.path.splitext(img_name)[0]}{aug_suffix}{os.path.splitext(img_name)[1]}")
            aug_img.save(aug_img_path)

        print(f"已處理圖片：{img_path}")

print("所有圖片擴增完成！")
