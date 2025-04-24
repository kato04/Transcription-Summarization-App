# %%writefile transcribe_app.py
# ↑ このマジックコマンドをセルの先頭に記述

# --- ここから下に提供されたコード全体をペースト ---
import streamlit as st
from pydub import AudioSegment
import whisper
import tempfile
import os
import math
import time
import traceback # エラー詳細表示用

# --- 定数 ---
CHUNK_LENGTH_MS = 10 * 60 * 1000 # 10分

# --- Whisperモデルのロード（キャッシュを利用して効率化）---
@st.cache_resource
def load_whisper_model(model_size="base"):
    # ...(以下、提供されたコードの通り)...
    st.info(f"Whisperモデル ({model_size}) をロード中...")
    try:
        model = whisper.load_model(model_size)
        st.success(f"Whisperモデル ({model_size}) のロード完了。")
        return model
    except Exception as e:
        st.error(f"Whisperモデルのロード中にエラーが発生しました: {e}")
        traceback.print_exc()
        return None

# --- アプリの UI 設定 ---
st.set_page_config(page_title="高精度文字起こしアプリ", layout="wide")
st.title("🚀 高精度文字起こしアプリ (Whisper)")
st.caption("M4Aファイルをアップロードすると、Whisperが自動で文字起こしします。長時間ファイルにも対応！")

# --- サイドバー: 設定 ---
st.sidebar.header("⚙️ 設定")
available_models = ["tiny", "base", "small", "medium"] # large はリソース的に厳しい可能性
default_model_index = available_models.index("base") if "base" in available_models else 0
model_size = st.sidebar.selectbox(
    "Whisperモデルを選択",
    available_models,
    index=default_model_index,
    help="モデルが大きいほど高精度ですが、処理時間とメモリ消費が増えます。'medium'以上はメモリ不足になる可能性あり。"
)

# --- メイン処理 ---
model = load_whisper_model(model_size)

if not model:
    st.warning("モデルのロードに失敗しました。")
else:
    uploaded_file = st.file_uploader(
        "M4Aファイルをここにドラッグ＆ドロップ または クリックして選択",
        type=['m4a'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        st.info(f"ファイル '{uploaded_file.name}' がアップロードされました。")
        st.audio(uploaded_file, format='audio/m4a')

        if st.button(f"文字起こし開始 ({model_size} モデル)", type="primary"):
            start_time = time.time()
            st.info("処理を開始します...")
            progress_bar = st.progress(0, text="準備中...")
            status_text = st.empty()

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    m4a_path = os.path.join(tmpdir, "uploaded.m4a")
                    wav_path = os.path.join(tmpdir, "converted.wav")
                    all_transcripts = []

                    status_text.text("ファイルを一時保存中...")
                    with open(m4a_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    progress_bar.progress(5, text="ファイルを一時保存中...")

                    status_text.text("WAV形式に変換中...")
                    progress_bar.progress(10, text="WAV形式に変換中...")
                    try:
                        audio = AudioSegment.from_file(m4a_path, format="m4a")
                        # WAVに変換する際にパラメータを指定（例: モノラル、16kHz）
                        # Whisperは16kHzモノラルを想定しているため、変換時に合わせると良い
                        audio = audio.set_channels(1).set_frame_rate(16000)
                        audio.export(wav_path, format="wav")
                        st.success("WAVファイルへの変換完了。")
                    except Exception as conversion_e:
                        st.error(f"音声ファイルの変換に失敗しました: {conversion_e}")
                        st.error("ffmpegが正しくインストールされているか、ファイル形式を確認してください。")
                        raise conversion_e # エラーを再送出して外側の try で捕捉

                    progress_bar.progress(15, text="文字起こし準備中...")
                    status_text.text("チャンク分割を準備中...")
                    wav_audio = AudioSegment.from_wav(wav_path)
                    total_length_ms = len(wav_audio)
                    num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS)

                    st.write(f"  音声の長さ: {total_length_ms / 1000:.1f} 秒")
                    st.write(f"  チャンク数: {num_chunks} (各チャンク最大 {CHUNK_LENGTH_MS / 60000} 分)")

                    if num_chunks == 0:
                         st.warning("音声ファイルが空か短すぎます。")
                         progress_bar.progress(100, text="完了（スキップ）")
                    else:
                        status_text.text(f"チャンク処理を開始... (全 {num_chunks} チャンク)")
                        for i in range(num_chunks):
                            chunk_start_time = time.time()
                            current_chunk_num = i + 1
                            progress_percentage = 15 + int((current_chunk_num / num_chunks) * 85)
                            progress_bar.progress(progress_percentage, text=f"チャンク {current_chunk_num}/{num_chunks} を処理中...")
                            status_text.text(f"チャンク {current_chunk_num}/{num_chunks} を処理中...")

                            start_ms = i * CHUNK_LENGTH_MS
                            end_ms = min((i + 1) * CHUNK_LENGTH_MS, total_length_ms)
                            chunk_audio = wav_audio[start_ms:end_ms]

                            chunk_wav_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                            chunk_audio.export(chunk_wav_path, format="wav")

                            # Whisperで文字起こし
                            # language="ja" を指定すると精度が上がる可能性がある
                            result = model.transcribe(chunk_wav_path, fp16=False, language="ja")
                            chunk_transcript = result["text"]
                            all_transcripts.append(chunk_transcript)

                            chunk_end_time = time.time()
                            print(f"Chunk {current_chunk_num}/{num_chunks} processed in {chunk_end_time - chunk_start_time:.2f} sec")

                        progress_bar.progress(100, text="文字起こし完了！")
                        status_text.success("全てのチャンクの文字起こしが完了しました！")

                        st.subheader("📝 文字起こし結果")
                        full_transcript = " ".join(all_transcripts).strip()
                        st.text_area("全文:", full_transcript, height=400)
                        st.download_button(
                            label="📄 結果をテキストファイルでダウンロード",
                            data=full_transcript.encode('utf-8'),
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt",
                            mime='text/plain',
                        )

                        # --- ここから要約機能を追加する場合 ---
                        # if st.button("要約を作成する (Gemini)"):
                        #    try:
                        #        # Geminiモデルの準備 (前のアプリのconfig.py参照)
                        #        # gemini_model = initialize_gemini() # APIキー設定などが必要
                        #        summary_prompt = f"以下の会議の文字起こしを要約してください:\n\n{full_transcript}"
                        #        # response = gemini_model.generate_content(summary_prompt)
                        #        # summary = response.text
                        #        # st.subheader("📜 要約結果")
                        #        # st.write(summary)
                        #        st.info("要約機能は現在準備中です。") # 仮置き
                        #    except Exception as summary_e:
                        #        st.error(f"要約の作成中にエラーが発生しました: {summary_e}")
                        # --- 要約機能ここまで ---

            except Exception as e:
                st.error("処理中に予期せぬエラーが発生しました。")
                st.error(f"エラー内容: {e}")
                st.code(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.progress(100, text="エラー発生")
                if 'status_text' in locals(): status_text.error("処理が中断されました。")

            finally:
                end_time = time.time()
                st.info(f"総処理時間: {end_time - start_time:.2f} 秒")

# --- ここまでメイン処理 ---