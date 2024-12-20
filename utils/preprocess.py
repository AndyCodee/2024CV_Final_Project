import cv2
import numpy as np

class Preprocess:
    def __init__(self):
        pass
    
    def hsv_segmentation(self, image):
        """
        將圖像轉換為HSV色彩空間並進行皮膚色分割。
        :param image: 原始BGR圖像
        :return: 皮膚色分割後的圖像
        """
        # 將圖像從BGR轉換為HSV色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定義更廣的皮膚色的HSV範圍
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)

        # 根據範圍創建遮罩
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 應用遮罩以獲取皮膚色區域
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def largest_connected_component(self, image):
        """
        保留圖像中最大的連通分量。
        :param image: 二值化的輸入圖像
        :return: 僅包含最大連通分量的二值化圖像
        """
        # 將圖像轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 應用二值化處理
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 進行連通分量標記
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 找到面積最大的連通分量（排除背景）
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # 創建最大的連通分量遮罩
        largest_component_mask = (labels == largest_label).astype("uint8") * 255
        
        # 應用遮罩以獲取最大的連通分量
        largest_component = cv2.bitwise_and(image, image, mask=largest_component_mask)
        
        return largest_component
    
    def gray_level(self, image):
        
        # 將圖像轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return gray


if __name__ == "__main__":
    # 測試用例：從文件讀取圖像，應用HSV分割，保留最大的連通分量，並顯示結果
    test_image_path = r'C:\Users\ouche\Desktop\CV_FP\dataset\label1\image_1.png'  # 替換為你的測試圖像路徑
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"無法讀取圖像：{test_image_path}")
        exit()

    # 創建Preprocess對象並應用HSV分割
    processor = Preprocess()
    
    segmented_image = processor.hsv_segmentation(image)
    largest_component_image = processor.largest_connected_component(segmented_image)
    gray_img = processor.gray_level(largest_component_image)
    
    # 顯示原始圖像和處理後的圖像
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.imshow('Largest Connected Component', largest_component_image)
    cv2.imshow('gray level', gray_img)
    
    # 等待按鍵按下，然後關閉所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
