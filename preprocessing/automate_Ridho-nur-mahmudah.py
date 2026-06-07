import pandas as pd
import numpy as np
import re
import string
import os
import csv
import requests
from io import StringIO
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def main():
    # --- A. KONFIGURASI PATH BERDASARKAN STRUKTUR WORKSPACE ---
    ROOT_DIR = 'Eksperimen_SML_Ridho-nur-mahmudah'
    INPUT_FILE = os.path.join(ROOT_DIR, 'dataset_raw', 'dataset_mobile_legends.csv')

    OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessing', 'dataset')
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

    # --- D. LOAD KAMUS LEXICON DARI GITHUB ---
    print("[*] Mengunduh kamus lexicon positif & negatif...")
    lexicon_positive = dict()
    res_pos = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
    if res_pos.status_code == 200:
        reader = csv.reader(StringIO(res_pos.text), delimiter=',')
        for row in reader: lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    res_neg = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
    if res_neg.status_code == 200:
        reader = csv.reader(StringIO(res_neg.text), delimiter=',')
        for row in reader: lexicon_negative[row[0]] = int(row[1])

    # --- E. SETUP STOPWORDS, STEMMER & SLANG DICTIONARY ---
    factory_sw = StopWordRemoverFactory()
    stop_words = set(factory_sw.get_stop_words())

    # PROTEKSI NEGASI: Keluarkan kata penting dari daftar hapus stopwords
    removable = {'tidak', 'kurang', 'lama', 'lambat', 'sulit', 'masalah', 'salah', 'bukan', 'jangan'}
    for w in removable:
        stop_words.discard(w)

    additional_sw = {
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'kalo', 'amp', 'biar', 'bikin', 'nya',
        'ini', 'itu', 'saya', 'dan', 'di', 'si', 'ya', 'aja', 'ke', 'ka', 'pun',
        'halo', 'admin', 'min', 'mohon', 'woy', 'sih', 'loh', 'user', 'url', 'kak', 'bang',
        'iya', 'yaa', 'gaa', 'woii', 'game', 'gamenya', 'moonton', 'ml', 'hero', 'rank'
    }
    stop_words.update(additional_sw)

    slang_dict = {
        "ga": "tidak", "gak": "tidak", "nggak": "tidak", "gk": "tidak", "tdk": "tidak", "ngak": "tidak",
        "bgt": "sekali", "banget": "sekali", "udah": "sudah", "sdh": "sudah", "dh": "sudah",
        "pake": "pakai", "matchmaking": "matchmaking", "macthmaking": "matchmaking",
        "lag": "lambat", "bug": "bug", "eror": "bug", "error": "bug", "lostrek": "lose streak",
        "jringan": "jaringan", "sinyal": "jaringan", "ping": "jaringan", "jlek": "jelek"
    }

    factory_stemmer = StemmerFactory()
    indonesian_stemmer = factory_stemmer.create_stemmer()

    # --- F. PIPELINE TEXT CLEANING ---
    def text_preprocessing_pipeline(text):
        if not isinstance(text, str):
            return ""

        # Cleaning regex standar teks
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'RT[\s]', '', text)
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = text.replace('\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text).lower().strip()

        # Pemetaan slang dan penyaringan kata (filtering)
        words = text.split()
        fixed_words = []
        for w in words:
            word = slang_dict[w] if w in slang_dict else w
            if word not in stop_words:
                fixed_words.append(word)

        return ' '.join(fixed_words)

    # --- G. FUNGSI PELABELAN LEXICON INDONESIA ---
    def lexicon_labeling(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0, 1
        words = text.split()
        score = 0
        for word in words:
            if word in lexicon_positive: score += abs(lexicon_positive[word])
            if word in lexicon_negative: score -= abs(lexicon_negative[word])

        if score > 0:
            return 2  # Positif
        elif score < 0:
            return 0  # Negatif
        else:
            return 1  # Netral

    # --- H. EKSEKUSI DATA PIPELINE ---
    print("[*] Memulai pembersihan teks, pelabelan, dan stemming (Mohon tunggu)...")
    if 'content' in df.columns:
        # 1. Transformasi Kolom Bersih
        df['content_clean'] = df['content'].apply(text_preprocessing_pipeline)
        df = df[df['content_clean'].str.strip() != ""].copy()

        # 2. Pelabelan Sentimen 3 Kelas (0, 1, 2)
        df['sentiment_label'] = df['content_clean'].apply(lexicon_labeling)

        # 3. Proses Stemming Kata Dasar Sastrawi
        df['content_stemmed'] = df['content_clean'].apply(indonesian_stemmer.stem)
    else:
        print("[!] Error: Kolom 'content' tidak ditemukan dalam dataset.")
        return

    # --- I. SIMPAN HASIL KE TARGET PREPROCESSED ---
    df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 65)
    print(f"[SUCCESS] File hasil otomatisasi disimpan di:\n  => {OUTPUT_FILE}")
    print(f"[*] Total baris data bersih akhir: {len(df)} baris")
    print(f"[*] Sebaran label otomatis: \n{df['sentiment_label'].value_counts()}")
    print("-" * 65)

if __name__ == "__main__":
    main()
