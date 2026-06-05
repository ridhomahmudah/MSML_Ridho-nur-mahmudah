import pandas as pd
import numpy as np
import re
import string
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import StandardScaler

def main():
    # --- A. CONFIGURASI PATH BERDASARKAN STRUKTUR WORKSPACE ---
    ROOT_DIR = 'Eksperimen_SML_Ridho-nur-mahmudah'
    INPUT_FILE = os.path.join(ROOT_DIR, 'dataset_raw', 'dataset_mobile_legends.csv')

    # Path folder output untuk namadataset_preprocessing
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessing', 'ndataset')
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dataset_mobile_legends_preprocessed.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- B. LOAD DATA ---
    try:
        if not os.path.exists(INPUT_FILE):
            print(f"[!] Error: File input {INPUT_FILE} tidak ditemukan.")
            return

        df = pd.read_csv(INPUT_FILE)
        print(f"[*] Berhasil memuat data mentah dari: {INPUT_FILE}")
    except Exception as e:
        print(f"[!] Error saat memuat file: {e}")
        return

    # --- C. DROP MISSING VALUES & DUPLICATES ---
    df = df.dropna(subset=['content']).copy()
    df = df.drop_duplicates(subset=['content'])

    # --- D. BINNING LABEL SENTIMEN ---
    def tentukan_sentimen(skor):
        if skor <= 2:
            return 0  # Negatif
        elif skor == 3:
            return 1  # Netral
        else:
            return 2  # Positif

    if 'score' in df.columns:
        df['sentiment_label'] = df['score'].apply(tentukan_sentimen)

    # --- E. SETUP STOPWORDS, STEMMER & SLANG DICTIONARY ---
    factory_sw = StopWordRemoverFactory()
    stop_words = set(factory_sw.get_stop_words())

    # Pertahankan kata negasi/krusial untuk analisis sentimen game
    removable = {'tidak', 'kurang', 'lama', 'lambat', 'sulit', 'masalah', 'salah'}
    for w in removable:
        stop_words.discard(w)

    additional_sw = {
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'kalo', 'amp', 'biar', 'bikin', 'nya',
        'ini', 'itu', 'saya', 'dan', 'di', 'si', 'ya', 'aja', 'ke', 'ka', 'pun',
        'halo', 'admin', 'min', 'mohon', 'woy', 'sih', 'loh', 'user', 'url', 'emel'
    }
    stop_words.update(additional_sw)

    slang_dict = {
        "ga": "tidak", "gak": "tidak", "nggak": "tidak", "gk": "tidak", "tdk": "tidak",
        "bgt": "sekali", "banget": "sekali", "udah": "sudah", "sdh": "sudah",
        "pake": "pakai", "macthmaking": "matchmaking", "machtmaking": "matchmaking",
        "lag": "lambat", "bug": "rusak"
    }

    factory_stemmer = StemmerFactory()
    indonesian_stemmer = factory_stemmer.create_stemmer()

    # --- F. PIPELINE PEMBERSIHAN TEKS & STEMMING ---
    def clean_and_stem_process(text):
        text = str(text).lower()
        # Menghapus URL, Angka, Tanda Baca, dan Karakter Non-Ascii
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation)).strip()
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Penanganan Slang & Pembuangan Stopwords
        words = text.split()
        fixed_words = []
        for w in words:
            word = slang_dict[w] if w in slang_dict else w
            if word not in stop_words and len(word) > 2:
                fixed_words.append(word)

        cleaned_text = ' '.join(fixed_words)

        # Stemming Sastrawi ke Kata Dasar
        return indonesian_stemmer.stem(cleaned_text)

    # --- G. EKSEKUSI PIPELINE ---
    print("[*] Memulai pembersihan teks dan stemming (Mohon tunggu)...")
    if 'content' in df.columns:
        df['content_clean'] = df['content'].apply(clean_and_stem_process)
        df = df[df['content_clean'].str.strip() != ""]
    else:
        print("[!] Error: Kolom 'content' tidak ditemukan dalam dataset.")
        return

    print("[*] Melakukan standarisasi fitur numerik ('thumbsUpCount')...")
    num_cols = ['thumbsUpCount']
    existing_num_cols = [col for col in num_cols if col in df.columns]

    if existing_num_cols:
        scaler = StandardScaler()
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols].astype(float))

    # --- H. SIMPAN HASIL KE TARGET PREPROCESSED ---
    df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 65)
    print(f"[SUCCESS] File hasil otomatisasi disimpan di:\n  => {OUTPUT_FILE}")
    print(f"[*] Total baris data bersih akhir: {len(df)} baris")
    print("-" * 65)

if __name__ == "__main__":
    main()
