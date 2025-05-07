from sklearn.svm import LinearSVC # LinearSVC'yi içeri aktarıyoruz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Modeli eğitme ve değerlendirme ana fonksiyonu
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Verilen eğitim verisi üzerinde LinearSVC modelini eğitir
    ve test verisi üzerinde performansını değerlendirir.
    """
    model = LinearSVC(max_iter=2000, dual=False, class_weight='balanced')

    # Modeli eğitim verisi üzerinde eğitme 
    print("\nModel Eğitiliyor (LinearSVC)...")
    model.fit(X_train, y_train)
    print("Model Eğitildi.")

    # Eğitilmiş model ile test verisi üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    print("\nModel Değerlendirme Sonuçları:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")

    # Precision, Recall, F1-Score
    # target_names: Etiketlerinizin listesi ('Negatif', 'Notr', 'Pozitif' gibi)
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=model.classes_))

    print("\nKarmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    return model # Eğitilmiş modeli döndürürülür