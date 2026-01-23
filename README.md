## Deney-1: Baseline FFP Koduna Dayalı Eğitim Stratejisi (Model Değiştirilmeden)

Bu repository’de gerçekleştirilen Deney-1, klasik **Future Frame Prediction (FFP)** tabanlı
anomali tespit yaklaşımını kullanan **ffp_baseline** kodu üzerine inşa edilmiştir.
Deneyin temel prensibi açık ve nettir:

> **Model mimarisi değiştirilmemiştir.**  
> Generator ve discriminator yapıları, giriş–çıkış formatı, inference (test) pipeline’ı
> ve PSNR tabanlı anomali skoru hesaplama yöntemi
> ffp_baseline ile birebir aynıdır.  
> Yapılan tüm değişiklikler yalnızca **eğitim süreci, loss kontrolü ve değerlendirme stratejisi**
> ile sınırlıdır.

Bu nedenle Deney-1:
- Yeni bir model veya mimari önermez,
- Baseline FFP’nin çekirdek çalışma mantığını bozmaz,
- **Model tasarımı değil, öğrenme stratejisi** odaklı bir deneydir.

---

## Baseline FFP Kodunun Kapsamı (Değişmeyen Kısım)

Klasik ffp_baseline yaklaşımı şu prensibe dayanır:

- Normal videolarda gelecek kare yüksek doğrulukla tahmin edilir,
- Anomali içeren videolarda bu tahmin hatası doğal olarak artar,
- Anomali skoru, doğrudan bu **tahmin hatası (PSNR)** üzerinden hesaplanır.

Bu nedenle baseline kod:
- Zamansal geçişleri doğrudan kısıtlayan bir **temporal consistency loss** içermez,
- Hareket alanında düzen zorlayan bir **motion loss** içermez,
- Zamanlar arası farkları açıkça modelleyen ek bir kayıp terimi kullanmaz.

Baseline FFP tasarımı bilinçli olarak sade tutulmuştur; çünkü güçlü zamansal veya hareket
kısıtları, anomali durumlarını da “düzgünleştirerek” ayırt edici hatayı bastırabilir
ve ROC-AUC performansını düşürebilir.

---

## Deney-1’de Baseline’a Eklenen Tüm Bileşenler

Deney-1 kapsamında, ffp_baseline koduna **modeli değiştirmeden** aşağıdaki eklemeler yapılmıştır.

### 1. Bezier Trajectory Loss (Baseline’da Yok)

Baseline FFP kodu, hareketi veya nesne yörüngelerini (trajectory) geometrik olarak
modellemez. Deney-1’de, normal sahnelerdeki hareketlerin daha **düzenli ve süreklilik
gösteren yörüngeler** izlediği varsayımından hareketle
**Bezier trajectory loss** eklenmiştir.

Bu loss’un amacı:
- Modele yeni bir görev tanımlamak değil,
- Normal davranışın zamansal ve geometrik düzenini
  regularizasyon yoluyla daha kararlı biçimde öğrenmesini sağlamaktır.

---

### 2. Temporal Consistency Loss (Baseline’da Yok)

Baseline FFP kodunda, ardışık kareler arasındaki zamansal tutarlılığı
doğrudan zorlayan bir loss bulunmamaktadır.

Deney-1’de eklenen **temporal consistency loss**:
- Tahmin edilen karelerin zaman içindeki geçişlerinde
  aşırı titreşimi azaltmayı,
- Ancak anomali sinyalini bastırmamayı hedefler.

Bu loss, model mimarisine müdahale etmez;
yalnızca öğrenme sürecinde düzenleyici (regularizer) rol üstlenir.

---

### 3. Motion Loss (Baseline’da Yok)

Baseline FFP yaklaşımı, hareket bilgisini doğrudan bir kayıp terimiyle
denetlemez.

Deney-1’de eklenen **motion loss**:
- Normal sahnelerdeki hareket düzeninin
  daha tutarlı öğrenilmesini amaçlar,
- Anomali sahnelerdeki düzensiz hareketlerin
  tahmin hatası olarak daha belirgin ortaya çıkmasını hedefler.

Bu loss da mimariyi değiştirmez;
sadece eğitim sürecini destekleyen bir regularizasyon terimidir.

---

### 4. Curriculum Learning (Gecikmeli ve Kontrollü Aktivasyon)

Deney-1’in en kritik farkı, eklenen loss’ların **eğitim başında agresif şekilde
kullanılmamasıdır**.

Bu kapsamda:
- Temporal, motion ve Bezier loss’lar eğitim başlangıcında kapalı tutulmuş,
- Model önce yalnızca baseline FFP kayıplarıyla temel gelecek kare tahmin
  davranışını öğrenmiştir,
- Bu ek loss’lar daha sonra **belirli epoch’lardan sonra**
  ve **düşük ağırlıklarla** devreye alınmıştır.

Bu yaklaşımın amacı:
- Erken aşamada aşırı regularizasyonu önlemek,
- Anomali sinyalinin bastırılmasını engellemek,
- ROC-AUC performansını olumsuz etkileyecek over-smoothing davranışını azaltmaktır.

---

### 5. AUC Odaklı Değerlendirme ve Checkpoint Seçimi

Baseline FFP uygulamalarında model seçimi sıklıkla
training loss veya PSNR değerlerine göre yapılır.

Deney-1’de ise:
- Anomali tespit probleminin asıl performans metriği olan
  **ROC-AUC** temel değerlendirme kriteri olarak alınmıştır,
- En iyi model checkpoint’i,
  en düşük loss’a değil,
  **en yüksek AUC değerine göre** seçilmiştir.

Bu yaklaşım, PSNR artmasına rağmen AUC’nin düşmesi gibi
yanıltıcı durumların önüne geçmiştir.

---

### 6. Geliştirilmiş Loglama ve Deney Görünürlüğü

Deney-1’de:
- Hangi loss’un hangi epoch’ta aktif olduğu,
- Loss ağırlıkları ve eğitim sürecindeki değişimler
  açık biçimde loglanmıştır.

Bu sayede:
- Eğitim süreci şeffaf hâle gelmiş,
- Deney sonuçları geriye dönük olarak
  bilimsel biçimde analiz edilebilir olmuştur.

---

## Sonuç

**Deney-1’de ffp_baseline koduna; temporal consistency loss, motion loss,
Bezier trajectory loss, curriculum learning, AUC bazlı checkpoint seçimi
ve gelişmiş loglama eklenmiştir.
Buna karşın model mimarisi, temel FFP pipeline’ı ve anomali skoru hesaplama
yöntemi tamamen korunmuştur.**

Bu deney, yeni bir model önermekten ziyade,
baseline FFP’nin öğrenme sürecinin doğru şekilde kontrol edilmesinin
ROC-AUC performansı üzerindeki etkisini incelemeyi amaçlamaktadır.

## Deney Sonuçları ve Google Drive Arşivi

Bu repository, kod ve deney konfigürasyonlarını içermektedir.
Deneyler sırasında üretilen **büyük boyutlu çıktılar** (model checkpoint’leri,
log dosyaları, ROC eğrileri ve skor dosyaları) repo boyutunu artırmamak ve
daha düzenli bir deney yönetimi sağlamak amacıyla **Google Drive** üzerinde
arşivlenmiştir.

### 📁 Google Drive Sonuç Arşivi
🔗 **Drive Linki:**  
(https://drive.google.com/drive/folders/1PDzWfFERlf3aZHVVAloE-PEGyxshbykH?usp=sharing)

---

## Drive İçeriği ve Dosyaların Anlamı

Drive klasörü, Deney-1’e ait tüm çıktıları aşağıdaki şekilde organize eder:

### 1️⃣ Model Checkpoint’leri
- `best_model_*.pth`  
  ROC-AUC değerine göre **en iyi performansı veren model ağırlıkları**.
- Ara checkpoint’ler (varsa):  
  Eğitim sürecindeki farklı epoch/iterasyonlara ait model kayıtları.

📌 Not:  
Checkpoint seçimi **training loss veya PSNR’ye göre değil**,  
doğrudan **ROC-AUC performansına göre** yapılmıştır.

---

### 2️⃣ Eğitim Logları
- Loss değerleri (reconstruction, temporal, motion, Bezier)
- Loss ağırlıkları (hangi epoch’ta aktif oldukları)
- PSNR değişimi
- Eğitim süresince izlenen temel metrikler

Bu loglar sayesinde:
- Eklenen loss’ların **ne zaman devreye girdiği**,
- Eğitim sürecini **nasıl etkilediği**
net biçimde izlenebilmektedir.

---

### 3️⃣ ROC Eğrileri ve AUC Sonuçları
- ROC curve görselleri (`.png` / `.pdf`)
- AUC skor dosyaları (`.txt` / `.csv`)

Bu çıktılar:
- Normal ve anomali sahnelerinin ne ölçüde ayrılabildiğini,
- Deney-1’in **baseline FFP’ye kıyasla AUC açısından nasıl davrandığını**
göstermek amacıyla saklanmaktadır.

---

### 4️⃣ Anomali Skorları ve Normalize Çıktılar
- Frame-bazlı anomali skorları
- PSNR tabanlı hata değerleri
- Normalize edilmiş skor dizileri

Bu dosyalar:
- ROC-AUC hesaplamasının nasıl yapıldığını,
- Deney sonuçlarının **tekrarlanabilir** olduğunu
kanıtlamak için paylaşılmaktadır.

---

## Sonuçların Genel Yorumu (Deney-1)

Google Drive üzerinde paylaşılan sonuçlar, Deney-1 kapsamında şu temel noktaları
göstermektedir:

- Model mimarisi değiştirilmeden,
- Temporal, motion ve Bezier loss’ların **curriculum learning ile kontrollü biçimde**
kullanılması,
- Normal sahnelerde tahmin davranışını daha kararlı hâle getirirken,
- Anomali sahnelerde tahmin hatasının daha ayırt edici olmasını sağlamıştır.

Bu durum, ROC-AUC metriğinde gözlemlenebilir bir iyileşme veya
en azından daha **kararlı ve yorumlanabilir** bir performans davranışı
elde edildiğini göstermektedir.

Drive arşivi, Deney-1 sonuçlarının:
- Şeffaf,
- İncelenebilir,
- Tekrar üretilebilir
olmasını sağlamak amacıyla paylaşılmıştır.

