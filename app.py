# app.py
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, AudioMessage, TextSendMessage

import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
import io
import soundfile as sf
import sounddevice as sd # 新增：用於麥克風錄音
import time
from datetime import datetime
from dotenv import load_dotenv
import warnings
import threading # 新增：用於多執行緒
import queue # 新增：用於執行緒間通信

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 載入 .env 檔案中的環境變數
load_dotenv()

# ==============================================================================
# --- Line Bot 設定 (從環境變數載入) ---
# ==============================================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("錯誤: LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET 未在 .env 檔案中設定。")
    exit()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = Flask(__name__)

# ==============================================================================
# --- 模型與 Scaler 路徑 ---
# ==============================================================================
MODEL_SAVE_PATH = 'best_baby_emotion_detector_cnn_model.h5'
SCALER_SAVE_PATH = 'mel_spectrogram_scaler.joblib'
USER_ID_FILE = 'line_user_id.txt' # 用於保存用戶ID的檔案

# ==============================================================================
# --- 模型訓練時使用的參數 (必須與訓練時一致) ---
# ==============================================================================
SR = 16000          # 樣本率 (Sample Rate)
N_FFT = 2048        # 傅立葉變換 (FFT) 的視窗大小
HOP_LENGTH = 512    # 幀之間的樣本數
N_MELS = 128        # 梅爾濾波器數量
TARGET_DURATION = 3 # 每段音訊的目標秒數 (模型訓練時的音訊長度)
MAX_FRAMES = int(np.ceil(TARGET_DURATION * SR / HOP_LENGTH))

CATEGORIES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'non_cry']
NUM_CLASSES = len(CATEGORIES)

# ==============================================================================
# --- 監測設定 ---
# ==============================================================================
RECORD_INTERVAL_SECONDS = 5 # 每隔多少秒進行一次偵測 (包含錄音和處理時間)
AUDIO_RECORD_DURATION = TARGET_DURATION # 每次錄音的時長

# ==============================================================================
# --- 全局變數和執行緒控制 ---
# ==============================================================================
monitor_thread = None
monitor_running = False
monitor_stop_event = threading.Event() # 用於發送停止信號
monitor_user_id = None # 用於儲存要推播的用戶ID

# ==============================================================================
# --- 載入模型和 Scaler (在應用程式啟動時載入一次) ---
# ==============================================================================
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    print("應用程式啟動: 模型和 Scaler 已成功載入。")
except Exception as e:
    print(f"應用程式啟動: 載入模型或 Scaler 失敗：{e}")
    print("請確保 'best_baby_emotion_detector_cnn_model.h5' 和 'mel_spectrogram_scaler.joblib' 存在於腳本同目錄。")
    exit()

# ==============================================================================
# --- 特徵提取函數 (從音訊陣列或 BytesIO) ---
# ==============================================================================
def extract_mel_spectrogram(audio_data, sr_input, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, target_duration=TARGET_DURATION):
    """
    從音訊數據陣列中提取梅爾頻譜圖。
    sr_input: 輸入音訊的採樣率
    """
    # 確保採樣率與模型訓練時一致
    if sr_input != SR:
        y = librosa.resample(y=audio_data, orig_sr=sr_input, target_sr=SR)
    else:
        y = audio_data

    target_samples = int(SR * target_duration) # 使用全局 SR
    
    if len(y) > target_samples:
        start_index = (len(y) - target_samples) // 2
        y = y[start_index : start_index + target_samples]
    elif len(y) < target_samples:
        padding = target_samples - len(y)
        y = np.pad(y, (0, padding), 'constant')

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    current_frames = mel_spectrogram_db.shape[1]
    if current_frames > MAX_FRAMES:
        mel_spectrogram_db = mel_spectrogram_db[:, :MAX_FRAMES]
    elif current_frames < MAX_FRAMES:
        padding = MAX_FRAMES - current_frames
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), 'constant')

    return mel_spectrogram_db

# ==============================================================================
# --- 預測函數 ---
# ==============================================================================
def predict_emotion(mel_spectrogram):
    """
    使用載入的模型預測梅爾頻譜圖的情緒。
    """
    try:
        original_shape = mel_spectrogram.shape
        mel_spectrogram_flat = mel_spectrogram.reshape(1, -1) 

        if not hasattr(scaler, 'mean_'):
            raise ValueError("Scaler 尚未擬合 (not fitted)。")

        mel_spectrogram_normalized_flat = scaler.transform(mel_spectrogram_flat)
        mel_spectrogram_normalized = mel_spectrogram_normalized_flat.reshape((1, original_shape[0], original_shape[1], 1))

        predictions = model.predict(mel_spectrogram_normalized, verbose=0)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index] * 100

        predicted_emotion = CATEGORIES[predicted_class_index]
        return predicted_emotion, confidence
    except Exception as e:
        print(f"預測時發生錯誤：{e}")
        return "錯誤", 0.0

# ==============================================================================
# --- 麥克風監測執行緒函數 ---
# ==============================================================================
def microphone_monitor_task(user_id, stop_event):
    global monitor_running
    monitor_running = True
    print(f"監測執行緒啟動，將推播至用戶 ID: {user_id}")
    
    try:
        line_bot_api.push_message(user_id, TextMessage(text="寶寶哭聲偵測器 (持續監測模式) 已啟動，每5秒將回傳結果。"))
    except Exception as e:
        print(f"監測執行緒：首次推播訊息失敗：{e}")

    while not stop_event.is_set(): # 檢查停止事件
        start_time = time.time()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 錄音 {AUDIO_RECORD_DURATION} 秒...")
        try:
            recording = sd.rec(int(AUDIO_RECORD_DURATION * SR), samplerate=SR, channels=1, dtype='float32')
            sd.wait() # 等待錄音完成
            audio_data = recording.flatten()
        except Exception as e:
            print(f"監測執行緒：錄音失敗：{e}。請檢查麥克風設定和驅動。")
            try:
                line_bot_api.push_message(user_id, TextMessage(text=f"錄音失敗：{e}。請檢查麥克風。"))
            except Exception as e_line:
                print(f"監測執行緒：推播 Line 訊息失敗：{e_line}")
            time.sleep(RECORD_INTERVAL_SECONDS) # 等待下次嘗試
            continue

        # 提取特徵並預測
        mel_spec = extract_mel_spectrogram(audio_data, SR) # 麥克風錄音的採樣率就是 SR
        if mel_spec is not None:
            predicted_emotion, confidence = predict_emotion(mel_spec)
            message = f"偵測到: {predicted_emotion} (信心度: {confidence:.2f}%)"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            
            # 推播結果到 Line
            try:
                line_bot_api.push_message(user_id, TextMessage(text=message))
            except Exception as e:
                print(f"監測執行緒：推播 Line 訊息失敗：{e}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 音訊處理失敗。")
            try:
                line_bot_api.push_message(user_id, TextMessage(text="音訊處理失敗，請重試。"))
            except Exception as e:
                print(f"監測執行緒：推播 Line 訊息失敗：{e}")

        # 計算已用時間，並等待到下一個 RECORD_INTERVAL_SECONDS 的邊界
        elapsed_time = time.time() - start_time
        sleep_time = RECORD_INTERVAL_SECONDS - elapsed_time
        if sleep_time > 0:
            # 在等待期間檢查停止事件，以便及時響應停止指令
            if stop_event.wait(sleep_time): # wait() 會在事件被設置時立即返回 True
                break # 如果停止事件被設置，則跳出循環

    monitor_running = False
    print(f"監測執行緒已停止。")
    try:
        line_bot_api.push_message(user_id, TextMessage(text="寶寶哭聲偵測器 (持續監測模式) 已停止。"))
    except Exception as e:
        print(f"監測執行緒：停止時推播 Line 訊息失敗：{e}")

# ==============================================================================
# --- Flask Webhook 路由 ---
# ==============================================================================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

# ==============================================================================
# --- 訊息處理器 ---
# ==============================================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    global monitor_thread, monitor_running, monitor_user_id

    user_id = event.source.user_id
    # 每次收到訊息都更新用戶ID，確保推播目標是最新的互動用戶
    monitor_user_id = user_id 
    # 將用戶 ID 保存到檔案，以便應用程式重啟後也能知道推播目標
    with open(USER_ID_FILE, 'w') as f:
        f.write(user_id)
    print(f"已將用戶 ID '{user_id}' 保存到 {USER_ID_FILE}")

    text = event.message.text.lower()
    reply_message = ""

    if "開始偵測" in text:
        if monitor_running:
            reply_message = "持續監測模式已經在運行中。"
        else:
            # 啟動監測執行緒
            monitor_stop_event.clear() # 清除停止信號
            monitor_thread = threading.Thread(target=microphone_monitor_task, args=(user_id, monitor_stop_event))
            monitor_thread.daemon = True # 設置為守護執行緒，主程式退出時會自動終止
            monitor_thread.start()
            reply_message = "持續監測模式已啟動。每5秒將回傳偵測結果。"
    elif "停止偵測" in text:
        if monitor_running:
            monitor_stop_event.set() # 發送停止信號
            # monitor_thread.join() # 不在這裡等待執行緒結束，避免阻塞 Webhook
            reply_message = "持續監測模式正在停止中，請稍候。"
        else:
            reply_message = "持續監測模式目前沒有運行。"
    elif "狀態" in text:
        status = "運行中" if monitor_running else "未運行"
        reply_message = f"持續監測模式目前狀態：{status}。"
    elif "你好" in text or "哈囉" in text:
        reply_message = "您好！我是寶寶哭聲偵測器。您可以發送 '開始偵測'、'停止偵測' 或 '狀態' 指令。\n" \
                        "您也可以直接傳送音訊檔案給我進行單次分析。"
    elif "幫助" in text or "help" in text:
        reply_message = "您可以發送以下指令：\n" \
                        "- '開始偵測': 啟動持續麥克風監測。\n" \
                        "- '停止偵測': 停止持續麥克風監測。\n" \
                        "- '狀態': 查詢監測模式狀態。\n" \
                        "您也可以直接發送寶寶的哭聲音訊給我進行單次分析。"
    else:
        reply_message = "抱歉，我不明白您的指令。請發送 '幫助' 查看可用指令。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_message)
    )

@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    global monitor_user_id
    user_id = event.source.user_id
    # 每次收到訊息都更新用戶ID
    monitor_user_id = user_id
    with open(USER_ID_FILE, 'w') as f:
        f.write(user_id)
    print(f"已將用戶 ID '{user_id}' 保存到 {USER_ID_FILE}")

    print(f"收到來自用戶 {event.source.user_id} 的音訊訊息 (單次分析)...")
    message_content = line_bot_api.get_message_content(event.message.id)
    
    audio_bytes_io = io.BytesIO(message_content.content)
    
    try:
        # 使用 soundfile 讀取 BytesIO 內容，並獲取其原始採樣率
        y, received_sr = sf.read(audio_bytes_io)
        # 執行特徵提取和預測
        mel_spec = extract_mel_spectrogram(y, received_sr)
        
        if mel_spec is not None:
            predicted_emotion, confidence = predict_emotion(mel_spec)
            reply_text = f"單次分析結果：偵測到 {predicted_emotion} (信心度: {confidence:.2f}%)"
        else:
            reply_text = "抱歉，無法處理您的音訊檔案，請確保音訊清晰或嘗試其他格式。"
    except Exception as e:
        print(f"處理音訊訊息時發生錯誤：{e}")
        reply_text = f"處理音訊時發生錯誤：{e}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# ==============================================================================
# --- 運行 Flask App ---
# ==============================================================================
if __name__ == "__main__":
    # 應用程式啟動時，嘗試載入上次保存的用戶ID
    if os.path.exists(USER_ID_FILE):
        with open(USER_ID_FILE, 'r') as f:
            monitor_user_id = f.read().strip()
            if monitor_user_id:
                print(f"應用程式啟動: 已載入上次的用戶 ID: {monitor_user_id}")
            else:
                print("應用程式啟動: line_user_id.txt 為空。")
    else:
        print("應用程式啟動: line_user_id.txt 不存在。")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)