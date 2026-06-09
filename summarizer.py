from transformers import pipeline
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg') # Mencegah error GUI matplotlib di server Flask
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
import translators as ts
import nltk
import os
import warnings

warnings.filterwarnings('ignore')

# Download library NLTK yang dibutuhkan jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Load BART Model (Diatur untuk GPU CUDA)
try:
    # device=0 routes the inference to the dedicated NVIDIA GPU
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    print("[OK] BART model loaded successfully on GPU")
except Exception as e:
    print(f"[WARN] Error loading BART model: {e}")
    summarizer = None

sentiment_analyzer = SentimentIntensityAnalyzer()

def text_summarize(text, method="bart", min_length=50, max_length=150):
    """Meringkas teks menggunakan BART atau algoritma Frekuensi"""
    if not text or len(text.strip()) < 50:
        return text
    
    # Deep Learning (BART)
    if method == "bart" and summarizer is not None:
        try:
            # Memotong teks jika terlalu panjang agar tidak crash (penting untuk limitasi VRAM)
            safe_text = text[:3500] 
            input_length = len(safe_text.split())
            adjusted_max = min(max_length, max(50, input_length // 2))
            adjusted_min = min(min_length, adjusted_max - 10)
                
            summary = summarizer(safe_text, max_length=adjusted_max, min_length=adjusted_min, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"BART failed: {e}. Menggunakan metode frekuensi.")
            
    # Fallback: Extractive (Word Frequency)
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    if not words: return text

    frequency_dist = FreqDist(words)
    max_freq = max(frequency_dist.values())
    sentence_scores = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency_dist.keys():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += frequency_dist[word] / max_freq
                else:
                    sentence_scores[sentence] = frequency_dist[word] / max_freq

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary_sentences.sort(key=lambda x: sentences.index(x))
    return " ".join(summary_sentences)

def translate_summary(text, target_lang='id'):
    """Machine Translation requirement (English to Indonesian)"""
    try:
        # Using translate_text from translators library (updated API)
        translation = ts.translate_text(text, from_language='en', to_language=target_lang)
        return translation
    except Exception as e:
        # Fallback: return original text if translation fails
        return text

def sentiment_analysis(text):
    """Menganalisis sentimen teks, returns dict with label and score"""
    analysis = TextBlob(text)
    sent_scores = sentiment_analyzer.polarity_scores(text)
    
    if analysis.sentiment.polarity > 0.05:
        label = "Positive"
    elif analysis.sentiment.polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    return {
        'label': label,
        'score': round(sent_scores['compound'], 4)
    }

def word_cloud(text, filename):
    """Membuat visualisasi kata (Word Cloud) dengan tema gelap"""
    fig = None
    try:
        # Custom colormap: cyan -> purple -> pink (matching UI gradient)
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#00d4ff', '#06b6d4', '#a855f7', '#7c3aed', '#ec4899', '#f43f5e']
        custom_cmap = LinearSegmentedColormap.from_list('nlp_gradient', colors, N=256)

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color='#0d0d1a',
            max_words=120,
            colormap=custom_cmap,
            contour_width=0,
            min_font_size=10,
            max_font_size=120,
            prefer_horizontal=0.7,
            relative_scaling=0.5
        ).generate(text)

        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor('#0d0d1a')
        ax = fig.add_subplot(111)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_facecolor('#0d0d1a')
        
        os.makedirs('static/images', exist_ok=True)
        file_path = os.path.join('static', 'images', filename)
        
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=120,
                    facecolor='#0d0d1a', edgecolor='none')
        print(f"[OK] Word Cloud saved to: {file_path}")
        return 'images/' + filename
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close('all')

def calculate_accuracy(ai_summary, csv_summary):
    """Menghitung seberapa mirip ringkasan AI dengan ringkasan asli Kaggle"""
    ai_words = set(ai_summary.lower().split())
    csv_words = set(csv_summary.lower().split())
    overlap = ai_words.intersection(csv_words)
    
    if not csv_words: return 0
    return round((len(overlap) / len(csv_words)) * 100, 2)