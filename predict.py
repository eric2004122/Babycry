# (這是 predict_baby_emotion_with_ui.py 的更新版本)
import librosa
import numpy as np
import tensorflow as tf
import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib # 用於載入 scaler

# 忽略 librosa 的未來警告
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. 定義參數和類別 (必須與訓練模型時完全一致！) ---
# 這些參數必須與訓練時使用的參數完全一致
SR = 16000 
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_DURATION = 3
MAX_FRAMES = int(np.ceil(TARGET_DURATION * SR / HOP_LENGTH)) 

# 類別順序必須與訓練時的 `categories` 列表完全一致
categories = ['belly_pain', 'burping', 'cry', 'discomfort', 'hungry', 'tired', 'non_cry']
num_classes = len(categories)

# --- 2. 定義特徵提取函數 (此函數必須與訓練時的 extract_mel_spectrogram 完全一致) ---
def extract_mel_spectrogram(audio_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, target_duration=TARGET_DURATION):
    """
    從音訊檔案中提取梅爾頻譜圖。
    注意：此函數的參數和處理邏輯必須與訓練時完全一致。
    預測時不應進行數據增強。
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        target_samples = int(sr * target_duration)
        if len(y) > target_samples:
            start_index = (len(y) - target_samples) // 2
            y = y[start_index : start_index + target_samples]
        elif len(y) < target_samples:
            padding = target_samples - len(y)
            y = np.pad(y, (0, padding), 'constant')

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        current_frames = mel_spectrogram_db.shape[1]
        if current_frames > MAX_FRAMES:
            mel_spectrogram_db = mel_spectrogram_db[:, :MAX_FRAMES]
        elif current_frames < MAX_FRAMES:
            padding = MAX_FRAMES - current_frames
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), 'constant')

        return mel_spectrogram_db # 返回未正規化的頻譜圖
    except Exception as e:
        print(f"處理檔案 {audio_path} 時發生錯誤：{e}")
        return None

# --- 3. 載入訓練好的模型和 Scaler ---
model_path = 'best_baby_emotion_detector_cnn_model.h5'
scaler_path = 'mel_spectrogram_scaler.joblib'

loaded_model = None
loaded_scaler = None

try:
    loaded_model = tf.keras.models.load_model(model_path)
    print("模型載入成功！")
except Exception as e:
    print(f"載入模型失敗：{e}")
    messagebox.showerror("模型載入失敗", f"無法載入模型檔案：'{model_path}'。\n請確認檔案是否存在於正確的路徑。")
    exit()

try:
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler 載入成功！")
except Exception as e:
    print(f"載入 Scaler 失敗：{e}")
    messagebox.showerror("Scaler 載入失敗", f"無法載入 Scaler 檔案：'{scaler_path}'。\n請確認檔案是否存在於正確的路徑。\n預測可能不準確。")
    # 不 exit，但會發出警告，因為沒有正確的正規化器
    loaded_scaler = None


# --- 4. 定義預測函數 ---
def predict_baby_emotion(audio_file_path, model, categories, scaler_obj):
    """
    對單個音訊檔案進行寶寶情緒預測。
    """
    if not os.path.exists(audio_file_path):
        return "錯誤：檔案不存在。", 0.0, None

    print(f"\n-----------------------------------------------------")
    print(f"正在預測檔案：{os.path.basename(audio_file_path)}")
    
    spectrogram = extract_mel_spectrogram(audio_file_path)
    
    if spectrogram is None:
        print("特徵提取失敗，無法進行預測。")
        return "特徵提取失敗。", 0.0, None

    if scaler_obj is None:
        print("警告：Scaler 未載入或載入失敗，將跳過正規化。預測結果可能不準確。")
        spectrogram_normalized = spectrogram # 如果沒有 scaler，則不正規化
    else:
        # 對提取的頻譜圖進行正規化 (使用訓練時保存的 scaler)
        # 將 2D 頻譜圖變平，進行 transform，再變回原始形狀
        spectrogram_normalized = scaler_obj.transform(spectrogram.reshape(1, -1)).reshape(spectrogram.shape)
    
    # 為 CNN 輸入添加批次維度 (batch dimension) 和通道維度
    input_data = spectrogram_normalized[np.newaxis, ..., np.newaxis]
    
    # 進行預測
    predictions = model.predict(input_data, verbose=0)[0] 
    
    predicted_class_index = np.argmax(predictions)
    predicted_category = categories[predicted_class_index]
    confidence = predictions[predicted_class_index]
    
    print(f"預測結果：'{predicted_category}' (信心程度: {confidence:.4f})")
    print("所有類別的機率分佈：")
    prediction_details = ""
    for i, prob in enumerate(predictions):
        print(f"  {categories[i]}: {prob:.4f}")
        prediction_details += f"{categories[i]}: {prob:.4f}\n" 
        
    return predicted_category, confidence, prediction_details

# --- 5. GUI 應用 (使用 Tkinter) ---
def open_file_dialog():
    root = tk.Tk()
    root.withdraw() 
    
    file_path = filedialog.askopenfilename(
        title="選擇寶寶音訊檔案 (.wav)",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    
    if file_path:
        predicted_category, confidence, details = predict_baby_emotion(file_path, loaded_model, categories, loaded_scaler)
        
        if predicted_category: 
            result_message = f"檔案: {os.path.basename(file_path)}\n\n" \
                             f"預測情緒: {predicted_category}\n" \
                             f"信心程度: {confidence:.4f}\n\n" \
                             f"詳細機率分佈:\n{details}"
            messagebox.showinfo("預測結果", result_message)
        else:
            messagebox.showerror("預測失敗", f"無法處理檔案：{os.path.basename(file_path)}\n錯誤訊息：{confidence}")
    else:
        messagebox.showinfo("取消", "未選擇任何檔案。")
    
    root.destroy() 

if __name__ == "__main__":
    print("準備就緒。將彈出檔案選擇對話框。")
    open_file_dialog()