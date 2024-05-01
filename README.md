# PBL#2 作業概述

## 簡介
這是一個機器視覺的作業，我是使用 cv2 還有搭配微調顏色、ROI、HSV等數值和遮罩來檢測正在打網球的球員。

## 畫面截圖
![image](https://github.com/YuCheng1122/PBL-2/assets/104905823/2cc0011a-5186-4cde-8e9d-04b3f7bba4e6)


## 設置
1. 複製檔案到本地端:
```bash
git clone <repository-url>
```
2. 安裝必要套件:
```bash
pip install requirements.txt
```

## 分析函數
### `get_dynamic_thresholds(y, height)`
- 根據每一偵的垂直位置確定動態逾值。

### `detect_players_optimized(video_path, speed_factor=1)`
以下是有關這次作業的步驟描述：
- 讀取指定的網球影片
- 定義 ROI（Region of Interest）參數
- 處理影片的每一幀
  1. 提取 ROI 並將其轉換為灰階和 HSV 
  2. 用於球員的顏色和運動軌跡檢測
  3. 對遮罩做清理
  4. 在 ROI 內找出遮罩中的輪廓
  5. 動態調整長寬比逾值
  6. 繪製檢測到球員周圍的邊框還有輸出分數
- 輸出檢測的log用於優化參數值和函數
- 按 'q' 程式停止運行


