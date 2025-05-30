import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib

# TensorFlow / Keras 相關庫
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# 忽略 librosa 的未來警告
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# --- 可調整參數配置 ---
# ==============================================================================

# --- 1. 資料路徑與類別定義 ---
DATA_DIR = r'C:/Babycry/Data' # 你的資料集根目錄
CATEGORIES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'non_cry'] # 定義所有情緒/聲音類別

# --- 2. 音訊處理與特徵提取參數 (梅爾頻譜圖) ---
SR = 16000          # 樣本率 (Sample Rate)
N_FFT = 2048        # 傅立葉變換 (FFT) 的視窗大小
HOP_LENGTH = 512    # 幀之間的樣本數 (決定時間解析度)
N_MELS = 128        # 梅爾濾波器數量 (決定頻率解析度，相當於圖像的高度)
TARGET_DURATION = 3 # 每段音訊的目標秒數 (建議3-5秒)
# 自動計算梅爾頻譜圖的寬度 (時間幀數)
MAX_FRAMES = int(np.ceil(TARGET_DURATION * SR / HOP_LENGTH))

# --- 3. 數據增強參數 ---
AUG_NOISE_FACTOR = 0.005 # 加入隨機噪音的強度
AUG_PITCH_FACTOR = 0.8   # 音高變化的最大步數
AUG_VOLUME_PROB = 0.5    # 調整音量的機率
AUG_NOISE_PROB = 0.5     # 加入噪音的機率
AUG_PITCH_PROB = 0.5     # 音高變化的機率

# --- 4. 模型架構參數 ---
# 卷積層配置: (濾波器數量, 卷積核大小, Dropout比率, L2正則化強度)
# 注意: Dropout比率和L2強度會應用到該層之後
CONV_LAYERS_CONFIG = [
    (64, (5, 5), 0.4, 0.0005),  # Conv1: filters, kernel_size, dropout_rate, l2_strength
    (128, (3, 3), 0.4, 0.0005), # Conv2
    (256, (3, 3), 0.5, 0.0005), # Conv3
    (512, (3, 3), 0.5, 0.0005)  # Conv4
]
POOL_SIZE = (2, 2) # 池化層大小 (所有MaxPooling2D層都使用此值)

# 全連接層配置: (神經元數量, Dropout比率, L2正則化強度)
DENSE_LAYER_CONFIG = (256, 0.6, 0.0005) # Dense1: units, dropout_rate, l2_strength

# --- 5. 訓練超參數 ---
INITIAL_LEARNING_RATE = 0.0001 # 初始學習率
EPOCHS = 50                    # 最大訓練 Epoch 數
BATCH_SIZE = 32                 # 批次大小
VALIDATION_SPLIT_RATIO = 0.1    # 從訓練集中分出作為驗證集的比例
TEST_SPLIT_RATIO = 0.2          # 測試集佔總數據的比例

# --- 6. 回調函數參數 ---
ES_PATIENCE = 20                # EarlyStopping 的 patience (在多少個 Epoch 內驗證損失沒有改善就停止)
LR_REDUCE_FACTOR = 0.5          # ReduceLROnPlateau 的 factor (學習率降低的倍數)
LR_REDUCE_PATIENCE = 5          # ReduceLROnPlateau 的 patience (在多少個 Epoch 內驗證損失沒有改善就降低學習率)
LR_MIN = 0.00001                # ReduceLROnPlateau 的最小學習率

# --- 7. 模型保存路徑 ---
MODEL_SAVE_PATH = 'best_baby_emotion_detector_cnn_model.h5'
SCALER_SAVE_PATH = 'mel_spectrogram_scaler.joblib'

# ==============================================================================
# --- 程式碼開始 ---
# ==============================================================================

# 為每個類別創建一個數字標籤映射
label_map = {category: i for i, category in enumerate(CATEGORIES)}
num_classes = len(CATEGORIES)

print(f"定義的類別及對應標籤：{label_map}")

# --- 1. 資料收集與準備 ---
audio_paths = []
labels = []

print("開始收集音訊檔案和標籤...")

# 收集所有定義類別的檔案
for category in CATEGORIES:
    category_path = os.path.join(DATA_DIR, category)
    if not os.path.exists(category_path):
        print(f"警告：未找到資料夾 '{category_path}'。請檢查路徑是否正確。此類別將被跳過。")
        continue

    for audio_file in glob.glob(os.path.join(category_path, '*.wav')):
        audio_paths.append(audio_file)
        labels.append(label_map[category]) # 使用數字標籤

if not audio_paths:
    print("錯誤：沒有找到任何音訊檔案。請檢查 DATA_DIR 和子資料夾內容。")
    exit()

print(f"總共收集到 {len(audio_paths)} 個音訊檔案。")


# --- 2. 音訊前處理 & 3. 特徵工程 (提取梅爾頻譜圖) ---

# 數據增強函數
def augment_audio(y, sr, noise_factor=AUG_NOISE_FACTOR, pitch_factor=AUG_PITCH_FACTOR):
    # 隨機調整音量
    if np.random.rand() < AUG_VOLUME_PROB:
        y = y * np.random.uniform(0.8, 1.2)

    # 加入隨機噪音
    if np.random.rand() < AUG_NOISE_PROB:
        noise = np.random.randn(len(y))
        y = y + noise_factor * noise

    # 音高變化
    if np.random.rand() < AUG_PITCH_PROB:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-pitch_factor, pitch_factor))
        
    return y

def extract_mel_spectrogram(audio_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, target_duration=TARGET_DURATION, augment=False):
    """
    從音訊檔案中提取梅爾頻譜圖。
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 進行數據增強 (只在訓練時使用)
        if augment:
            y = augment_audio(y, sr)
        
        # 處理音訊長度：確保所有音訊片段長度一致
        target_samples = int(sr * target_duration)
        
        if len(y) > target_samples:
            # 如果音訊太長，則從中間截取
            start_index = (len(y) - target_samples) // 2
            y = y[start_index : start_index + target_samples]
        elif len(y) < target_samples:
            # 如果音訊太短，則用零填充
            padding = target_samples - len(y)
            y = np.pad(y, (0, padding), 'constant')

        # 提取梅爾頻譜圖
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        # 轉換為分貝刻度 (Log-Mel Spectrogram)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 確保頻譜圖的寬度 (時間幀數) 與 MAX_FRAMES 相同
        current_frames = mel_spectrogram_db.shape[1]
        if current_frames > MAX_FRAMES:
            mel_spectrogram_db = mel_spectrogram_db[:, :MAX_FRAMES]
        elif current_frames < MAX_FRAMES:
            padding = MAX_FRAMES - current_frames
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), 'constant')

        return mel_spectrogram_db # 暫時不正規化，稍後進行全局正規化
    except Exception as e:
        print(f"處理檔案 {audio_path} 時發生錯誤：{e}")
        return None

X_spectrograms = []
y_labels_raw = []

print(f"開始提取梅爾頻譜圖 (維度: {N_MELS}x{MAX_FRAMES}, 可能需要一些時間)...")
print(f"將對每個樣本進行數據增強，創建 {len(audio_paths) * 2} 個有效樣本。") 

for i, path in enumerate(audio_paths):
    # 提取原始樣本的頻譜圖
    spec_orig = extract_mel_spectrogram(path, augment=False)
    if spec_orig is not None:
        X_spectrograms.append(spec_orig)
        y_labels_raw.append(labels[i])
    
    # 提取增強樣本的頻譜圖
    spec_aug = extract_mel_spectrogram(path, augment=True)
    if spec_aug is not None:
        X_spectrograms.append(spec_aug)
        y_labels_raw.append(labels[i]) # 增強樣本的標籤與原始樣本相同
    
    if (i + 1) % 50 == 0:
        print(f"已處理 {i + 1}/{len(audio_paths)} 個原始檔案...")

X_spectrograms = np.array(X_spectrograms)
y_labels_raw = np.array(y_labels_raw)

print(f"原始特徵提取完成。總樣本數：{X_spectrograms.shape[0]}，頻譜圖形狀：{X_spectrograms.shape[1:]}")

# --- 全局正規化 (Z-score Normalization) ---
# 在分割數據集之前，對所有頻譜圖進行全局正規化
# 這樣訓練集、驗證集和測試集都使用相同的統計量
scaler = StandardScaler()
# 將 3D 頻譜圖數據展平為 2D 進行 fit_transform，然後再變回 3D
original_shape = X_spectrograms.shape
X_spectrograms_flat = X_spectrograms.reshape(original_shape[0], -1)
X_spectrograms_normalized = scaler.fit_transform(X_spectrograms_flat)
X_spectrograms_normalized = X_spectrograms_normalized.reshape(original_shape)

# 保存 scaler 物件，以便在預測時使用相同的正規化參數
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"正規化器 (Scaler) 已保存至: {SCALER_SAVE_PATH}")

# 將數字標籤轉換為 One-Hot 編碼
y_labels = to_categorical(y_labels_raw, num_classes=num_classes)

# 為 CNN 輸入添加通道維度 (因為是灰度圖，所以通道數為 1)
X_spectrograms_normalized = X_spectrograms_normalized[..., np.newaxis] 

print(f"特徵提取和正規化完成。總樣本數：{X_spectrograms_normalized.shape[0]}，頻譜圖形狀：{X_spectrograms_normalized.shape[1:]}")
print(f"One-Hot 編碼後的標籤形狀：{y_labels.shape}")


# --- 4. 模型選擇與訓練 (使用 CNN) ---
# 分割資料集為訓練集和測試集
# stratify 應基於原始標籤，確保類別分佈均勻
X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X_spectrograms_normalized, y_labels, y_labels_raw, test_size=TEST_SPLIT_RATIO, random_state=42, stratify=y_labels_raw
)

print(f"\n訓練集大小：{X_train.shape[0]} 樣本")
print(f"測試集大小：{X_test.shape[0]} 樣本")

# --- 類別平衡處理 (過採樣) ---
print("\n--- 開始對訓練集進行類別平衡 (過採樣) ---")
unique_classes_train, class_counts_train = np.unique(y_train_raw, return_counts=True)
print(f"訓練集原始類別分佈 (原始標籤): {dict(zip(CATEGORIES, np.bincount(y_train_raw)))}")

max_samples = np.max(class_counts_train)
X_train_balanced = X_train.copy()
y_train_balanced = y_train.copy()
y_train_raw_balanced = y_train_raw.copy() # 也複製原始標籤用於後續檢查

for i, count in zip(unique_classes_train, class_counts_train):
    if count < max_samples:
        samples_to_add = max_samples - count
        # 找到該類別的索引
        class_indices = np.where(y_train_raw == i)[0]
        # 從該類別中隨機選擇樣本進行複製 (允許重複選擇)
        oversample_indices = np.random.choice(class_indices, samples_to_add, replace=True)
        
        X_train_balanced = np.concatenate((X_train_balanced, X_train[oversample_indices]), axis=0)
        y_train_balanced = np.concatenate((y_train_balanced, y_train[oversample_indices]), axis=0)
        y_train_raw_balanced = np.concatenate((y_train_raw_balanced, y_train_raw[oversample_indices]), axis=0)

# 打亂平衡後的數據集，防止模型學習到過採樣的順序模式
permutation = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[permutation]
y_train_balanced = y_train_balanced[permutation]
y_train_raw_balanced = y_train_raw_balanced[permutation] # 保持原始標籤的順序一致

print(f"訓練集平衡後總樣本數：{X_train_balanced.shape[0]}")
print(f"訓練集平衡後類別分佈 (原始標籤): {dict(zip(CATEGORIES, np.bincount(y_train_raw_balanced)))}")
print("--- 類別平衡處理完成 ---")


# 建立 CNN 模型
model = Sequential()

# 動態添加卷積層
for i, (filters, kernel_size, dropout_rate, l2_strength) in enumerate(CONV_LAYERS_CONFIG):
    if i == 0: # 第一層需要指定 input_shape
        model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=(N_MELS, MAX_FRAMES, 1), padding='same',
                         kernel_regularizer=regularizers.l2(l2_strength)))
    else:
        model.add(Conv2D(filters, kernel_size, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(POOL_SIZE))
    model.add(Dropout(dropout_rate))
    
model.add(GlobalAveragePooling2D()) # 使用全局平均池化取代 Flatten

# 添加全連接層
dense_units, dense_dropout_rate, dense_l2_strength = DENSE_LAYER_CONFIG
model.add(Dense(dense_units, activation='relu',
                kernel_regularizer=regularizers.l2(dense_l2_strength)))
model.add(BatchNormalization())
model.add(Dropout(dense_dropout_rate))

# 輸出層
model.add(Dense(num_classes, activation='softmax')) 

# 編譯模型
model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE), 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 

model.summary() 

# 定義回調函數
early_stopping = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, 
                                   monitor='val_loss', 
                                   save_best_only=True, 
                                   verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN, verbose=1)

# 計算類別權重 (處理類別不平衡問題)
# 注意：這裡的 class_weights 仍然基於原始的 y_train_raw 計算，
# 因為過採樣已經在物理上平衡了數據量，但 class_weights 可以在損失函數層面提供額外調整。
# 如果過採樣後數據完全平衡，這些權重將都接近 1.0。
unique_classes_for_weights, class_counts_for_weights = np.unique(y_train_raw, return_counts=True) # 使用原始訓練集的統計量
total_samples_for_weights = len(y_train_raw) # <--- 修正：在這裡定義 total_samples_for_weights

class_weights = {}
for i, count in zip(unique_classes_for_weights, class_counts_for_weights):
    class_weights[i] = total_samples_for_weights / (num_classes * count) 

print(f"使用的類別權重 (基於原始訓練集分佈): {class_weights}")


history = model.fit(
    X_train_balanced, y_train_balanced, # <--- 使用平衡後的訓練數據
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT_RATIO,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    class_weight=class_weights # 仍然使用 class_weight，即使過採樣了，也可以作為微調
)
print("模型訓練完成。")

# --- 打印最終訓練和驗證的損失和準確率 ---
print("\n--- 最終訓練結果 ---")
if history.history['loss']:
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1] 

    print(f"最後一個 Epoch 的訓練損失: {final_train_loss:.4f}")
    print(f"最後一個 Epoch 的訓練準確率: {final_train_acc:.4f}")
    print(f"最後一個 Epoch 的驗證損失: {final_val_loss:.4f}")
    print(f"最後一個 Epoch 的驗證準確率: {final_val_acc:.4f}")
    print("\n(注意: EarlyStopping 會恢復模型到驗證損失最低的 Epoch 的權重，上述為 '最後一個訓練 Epoch' 的性能)")
else:
    print("訓練歷史數據不足，無法打印最後一個 Epoch 的性能。")


# --- 5. 模型評估 ---
print("\n--- 模型評估 ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"測試集損失: {loss:.4f}")
print(f"測試集準確率: {accuracy:.4f}")

# 預測測試集
y_pred_proba = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1) # 獲取預測的類別索引

# 將真實標籤從 One-Hot 編碼轉換回數字索引
y_true_classes = np.argmax(y_test, axis=1) 

# 打印分類報告
print("\n分類報告:")
print(classification_report(y_true_classes, y_pred_classes, target_names=CATEGORIES))

# 打印混淆矩陣
print("\n混淆矩陣 (Confusion Matrix):")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# 可視化混淆矩陣
plt.figure(figsize=(num_classes + 2, num_classes + 1)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('預測標籤 (Predicted Label)')
plt.ylabel('真實標籤 (True Label)')
plt.title('混淆矩陣 (Confusion Matrix)')
plt.show()

# 可視化訓練過程中的損失和準確率
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()