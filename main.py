from process_data import load_process_and_vectorize_data
from train_model import train_and_evaluate_model
import os # Dosya yolu işlemleri için

if __name__ == "__main__":

    labeled_file_name = 'yorumlar_etiketlenecek.csv'
    labeled_file_path = os.path.join(os.path.dirname(__file__), labeled_file_name)

    # Veri setindeki yorum metni ve duygu etiketi sütunlarının isimlerini belirtin
    review_col = 'reviewText' # Yorum metni sütununun adı
    sentiment_col = 'Sentiment' # Duygu etiketi sütununun adı

    # Veriyi yükleme, işleme, ayırma ve vektörleştirme
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = load_process_and_vectorize_data(
        file_path=labeled_file_path,
        review_text_col=review_col,
        sentiment_col=sentiment_col,
        sample_size=None # Etiketlediğiniz tüm veriyi kullan
    )

    if X_train_tfidf is not None:
        # Modeli eğitme ve değerlendirme
        trained_model = train_and_evaluate_model(
            X_train=X_train_tfidf,
            X_test=X_test_tfidf,
            y_train=y_train,
            y_test=y_test
        )
    else:
        print("\nVeri yükleme veya işleme hatası nedeniyle model eğitilemedi.")