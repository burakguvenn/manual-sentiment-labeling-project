import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Metin Temizleme Fonksiyonu ---
def clean_text(text):
    """
    Metinleri küçük harfe çevirir, URL, özel karakter ve sayıları kaldırır,
    fazla boşlukları düzenler.
    """
    # string olduğundan emin olun, None veya NaN gelirse boş string yap
    if not isinstance(text, str):
        return ''

    text = text.lower() # Küçük harfe çevir
    text = re.sub(r'http\S+', '', text) # URL'leri kaldır
    text = re.sub(r'[^a-z\s]', '', text) # Harf ve boşluk dışındaki karakterleri kaldır (İngilizce harfler için)
    text = re.sub(r'\s+', ' ', text).strip() # Birden fazla boşluğu tek boşluğa indir ve baştaki/sondaki boşlukları kaldır

    return text

# --- Veri İşleme ve Vektörleştirme Fonksiyonu ---
def load_process_and_vectorize_data(file_path, review_text_col, sentiment_col, sample_size=None, test_size=0.2, random_state=42):
    """
    CSV dosyasını yükler, etiketleri temizler/eşler (ikili sınıflama için),
    yorum metinlerini temizler, eğitim/test setlerine ayırır
    ve TF-IDF vektörleştirmesi yapar.

    Args:
        file_path (str): Etiketlenmiş CSV dosyasının yolu.
        review_text_col (str): Yorum metnini içeren sütunun adı.
        sentiment_col (str): Duygu etiketini içeren sütunun adı.
        sample_size (int, optional): Eğer None değilse, veri setinden alınacak örneklem boyutu.
        test_size (float, optional): Test seti için veri setinin oranı (0.0 ile 1.0 arası). Varsayılan 0.2.
        random_state (int, optional): Veri ayırma ve örnekleme için rastgelelik durumu. Varsayılan 42.

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer)
               veya hata durumunda (None, None, None, None, None).
    """
    # Veriyi yükleme
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı. Lütfen yolu kontrol edin: {file_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        print("Farklı encoding denemeyi düşünebilirsiniz. Örn: encoding='latin1'")
        return None, None, None, None, None

    # Gerekli sütunların varlığını kontrol etme
    if review_text_col not in df.columns or sentiment_col not in df.columns:
        print(f"Hata: '{review_text_col}' veya '{sentiment_col}' sütunları dosyada bulunamadı.")
        print("Mevcut sütunlar:", df.columns.tolist())
        return None, None, None, None, None

    # Sadece gerekli sütunları alma
    data = df[[review_text_col, sentiment_col]].copy()

    # Bu sütunlardaki eksik değerleri atma (etiketleme dosyasında tam olması beklenir)
    initial_rows_before_dropna = len(data)
    data.dropna(subset=[review_text_col, sentiment_col], inplace=True)
    if len(data) < initial_rows_before_dropna:
         print(f"Uyarı: {initial_rows_before_dropna - len(data)} satır eksik metin veya etiket nedeniyle çıkarıldı.")


    # Eğer örneklem boyutu belirtilmişse ve veri setinden küçükse rastgele örneklem al
    if sample_size is not None and len(data) > sample_size:
         data = data.sample(n=sample_size, random_state=random_state).copy()
         print(f"Belirtilen örneklem boyutu ({sample_size}) kadar veri alındı.")
    else:
         print(f"Tüm kullanılabilir veri ({len(data)}) kullanılacak.")


    # --- Etiket sütunundaki olası hataları temizleme ve İKİLİ SINIFLANDIRMA için etiketleri eşleme ---
    print(f"Etiket sütunu ('{sentiment_col}') temizleniyor ve 'Pozitif'/'Olumsuz' olarak eşleniyor...")
    # Etiketleri küçük harfe çevir ve baştaki/sondaki boşlukları sil
    data[sentiment_col] = data[sentiment_col].astype(str).str.lower().str.strip()

    # İkili sınıflandırma için etiket eşleme sözlüğü:
    # Anahtarlar (sol taraf) etiketleme dosyanızdaki Orijinal etiketler (küçük harf/boşluksuz halleri)
    # Değerler (sağ taraf) kullanmak istediğiniz Nihai etiketler ('Pozitif', 'Olumsuz')
    label_mapping = {
        'pozitif': 'Pozitif',
        'negatif': 'Olumsuz', # Negatifleri 'Olumsuz' yap
        'nötr': 'Olumsuz',     # Nötrleri (eski yazım veya encoding hatalısı) 'Olumsuz' yap
        'nï¿½tr': 'Olumsuz', # Olası encoding hatası
        'notr': 'Olumsuz',   # Sizin CSV'de kullandığınız yazım
    }

    # Map fonksiyonu ile etiketleri değiştirin.
    # Eğer bir etiketin karşılığı map'te yoksa, o satırın 'cleaned_sentiment' değeri NaN olacaktır.
    # Bu genellikle etiketleme sırasında map'te olmayan bir değer girildiği anlamına gelir.
    data['cleaned_sentiment'] = data[sentiment_col].map(label_mapping)

    # Mapleme sonrası hala NaN (eşlenememiş etiket) kalan satırları çıkarma
    # Bu adım, map'te olmayan etiketleri atmaya yarar.
    initial_rows_after_map = len(data)
    data.dropna(subset=['cleaned_sentiment'], inplace=True)
    if len(data) < initial_rows_after_map:
         print(f"Uyarı: {initial_rows_after_map - len(data)} satır eşlenemeyen etiket nedeniyle çıkarıldı.")

    print("Temizlenmiş İkili Etiket Dağılımı:")
    print(data['cleaned_sentiment'].value_counts())

    print("Yorum metinleri temizleniyor...")
    #clean_text fonksiyonunu Orijinal yorum metni sütununa uygulanır ('review_text_col')
    data['cleaned_review'] = data[review_text_col].apply(clean_text)

    # Bağımlı (target) ve bağımsız (features) değişkenleri belirleme
    # Bağımsız değişken: Temizlenmiş yorum metinleri
    X = data['cleaned_review']
    # Bağımlı değişken: Temizlenmiş ve ikili sınıflandırmaya eşlenmiş etiketler
    y = data['cleaned_sentiment']

    # Veri setini eğitim ve test setlerine ayırma
    print("Veri eğitim ve test setlerine ayrılıyor...")
    # stratify=y parametresi, etiket dağılımının eğitim ve test setlerinde benzer olmasını sağlar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # --- TF-IDF Vektörleştirmesi ---
    print("Metinler TF-IDF ile vektörleştiriliyor...")
    # max_features: Vektörde kullanılacak en sık geçen kelime/ngram sayısı
    # min_df: Bir kelime/ngramın dikkate alınması için geçmesi gereken minimum belge sayısı oranı veya sayısı (örneğin 5 -> en az 5 yorumda geçmeli)
    # ngram_range: Kullanılacak n-gramların boyutu (1,1 -> sadece tek kelimeler, (1,2) -> tek kelimeler ve ikili kelime grupları)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2)) 

    # Vektörleyiciyi SADECE eğitim verisi üzerinde eğitin ve hem eğitim hem de test verisine uygulayın
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # test verisini sadece transform yapın, fit yapmayın!
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"Eğitim veri matrisinin boyutu: {X_train_tfidf.shape}")
    print(f"Test veri matrisinin boyutu: {X_test_tfidf.shape}")

    # Eğitilmiş modeli ve vektörleyiciyi kaydetmek icin döndürülür
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer
