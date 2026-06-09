from flask import Flask, render_template, request
from summarizer import text_summarize, sentiment_analysis, word_cloud, translate_summary, calculate_accuracy
from csvloader import CSVLoader
import os
import glob
import time
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Membuat folder yang dibutuhkan secara otomatis
os.makedirs('static/images', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def cleanup_old_wordclouds(keep_latest=5):
    """Hapus wordcloud lama untuk menghemat disk space"""
    try:
        pattern = os.path.join('static', 'images', 'wordcloud_*.png')
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old_file in files[keep_latest:]:
            os.remove(old_file)
    except Exception:
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    try:
        start_time = time.time()

        csv_file = request.files.get('csv_file')
        row_index = int(request.form.get('row_index', 0))

        if not csv_file or csv_file.filename == '':
            return render_template('result.html', display="Error",
                                   result="Silakan upload file test.csv dari Kaggle")

        # 1. Simpan CSV yang diupload
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
        csv_file.save(file_path)

        # 2. Load Data menggunakan CSVLoader
        loader = CSVLoader(file_path)
        article, original_summary = loader.get_article_summary_pair(row_index)

        if not article:
            return render_template('result.html', display="Error",
                                   result="Artikel kosong pada baris tersebut.")

        # 3. Proses NLP Pipeline
        # Summarization (Menggunakan BART)
        ai_summary = text_summarize(article, method="bart")

        # Machine Translation ke Bahasa Indonesia
        translation = translate_summary(ai_summary, target_lang='id')

        # Sentiment Analysis (returns dict with 'label' and 'score')
        sentiment = sentiment_analysis(article)

        # Validasi Akurasi (Kaggle dataset comparison)
        accuracy_score = calculate_accuracy(ai_summary, original_summary)

        # Generate Word Cloud
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wc_filename = f"wordcloud_{timestamp}.png"
        wc_result = word_cloud(article, wc_filename)

        # Hapus file CSV setelah selesai dibaca untuk menghemat space
        os.remove(file_path)

        # Cleanup old wordcloud images
        cleanup_old_wordclouds(keep_latest=5)

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)

        # 4. Kirim hasil ke Dashboard HTML
        return render_template('result.html',
                               article=article[:500] + "...",
                               original_summary=original_summary,
                               ai_summary=ai_summary,
                               translation=translation,
                               sentiment_label=sentiment['label'],
                               sentiment_score=sentiment['score'],
                               accuracy=accuracy_score,
                               image_url=wc_result,
                               processing_time=processing_time)

    except Exception as e:
        error_msg = f"Terjadi kesalahan: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return render_template('result.html', display="Error", result=error_msg)


if __name__ == '__main__':
    print("Starting Complete NLP Dashboard Engine")
    print("Menjalankan server lokal di http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)