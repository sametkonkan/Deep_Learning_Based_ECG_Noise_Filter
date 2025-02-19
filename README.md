# ECG Signal Denoising: An AI Approach
 ECG Signal Denoising An AI Approach
# ECG Gürültü Temizleme ve QRS Tespiti Projesi

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📜 Proje Amacı
MIT-BIH Aritmi Veritabanı kullanılarak EKG sinyallerinden:
- **Gürültü temizleme** (Bazal sürüklenme, EMG, hareket artefaktı)
- **Wavelet & Derin Öğrenme** tabanlı hibrit filtreleme
- **Pan-Tompkins & Hilbert** algoritmaları ile QRS tespiti
- Performans analizi (SNR, MSE, Sensitivite)

## 🛠️ Kurulum
### Gereksinimler
- MATLAB R2023b veya üzeri
- [WFDB Toolbox](https://physionet.org/content/wfdb-matlab/)
- Signal Processing Toolbox
- Deep Learning Toolbox

### Adımlar
1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici_adiniz/ecg-denoising.git
   ```
2. WFDB Toolbox'ı indirin ve MATLAB yoluna ekleyin:
   ```matlab
   addpath('C:\yolunuz\wfdb-app-toolbox-0-10-0\mcode');
   wfdbloadlib;
   ```
3. MIT-BIH verilerini [PhysioNet](https://physio.org/content/mitdb/)'ten indirip `data/` klasörüne yerleştirin.

## 🚀 Kullanım
1. Ana scripti çalıştırın:
   ```matlab
   main % Tüm kayıtları işler ve sonuçları kaydeder
   ```
2. Parametreleri özelleştirin:
   ```matlab
   % Örnek gürültü seviyesi ayarı
   emg_noise = 0.4 * randn(size(ecg_clean)); 
   ```
3. Eğitilmiş yapay zeka modeli yükleyin:
   ```matlab
   load('ecg_denoiser_net.mat'); 
   ```

## 📊 Sonuçlar
### Örnek Çıktılar
![ECG Analiz]([ECG_Analysis_100.png](https://github.com/sametkonkan/Deep_Learning_Based_ECG_Noise_Filter/blob/main/results/ECG_Analysis_104.png))

### Performans Metrikleri
| Metrik       | Wavelet | Derin Öğrenme |
|--------------|---------|---------------|
| SNR (dB)     | 12.4    | 14.2          |
| MSE          | 0.032   | 0.028         |
| Sensitivite  | 92.3%   | 89.7%         |

## 📚 Referanslar
1. [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
2. [WFDB Toolbox for MATLAB](https://physionet.org/content/wfdb-matlab/)
3. Pan, J., & Tompkins, W. J. (1985). [A Real-Time QRS Detection Algorithm](https://ieeexplore.ieee.org/document/4122029)
4. Mallat, S. (1999). [A Wavelet Tour of Signal Processing](https://dl.acm.org/doi/book/10.5555/553543)

## 📧 İletişim
Sorularınız için: https://www.linkedin.com/in/samet-konkan/
