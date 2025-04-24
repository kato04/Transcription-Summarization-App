import streamlit as st
from google.cloud import speech
from google.oauth2 import service_account
import json
import io # ファイルライクオブジェクトを扱うために必要になる場合がある

# ----- アプリのタイトル -----
st.title("音声認識アプリ by Google Cloud STT")
st.write("音声ファイルをアップロードすると文字起こし結果を表示します。")

# ----- 認証情報の設定 (Streamlit Secrets から読み込み) -----
try:
    # Streamlit Cloud の Secrets から JSON 文字列を取得
    google_credentials_json_str = st.secrets["google_credentials_json"]

    # JSON 文字列を辞書に変換
    google_credentials_dict = json.loads(google_credentials_json_str)

    # 辞書から認証情報オブジェクトを作成
    credentials = service_account.Credentials.from_service_account_info(google_credentials_dict)

    # 認証情報を使って Speech-to-Text クライアントを初期化
    client = speech.SpeechClient(credentials=credentials)

    st.success("Google Cloud への認証に成功しました。")

    # ----- 音声ファイルのアップロード -----
    uploaded_file = st.file_uploader(
        "文字起こししたい音声ファイルを選択してください",
        type=["wav", "flac", "mp3", "ogg", "m4a", "opus", "amr"] # 対応している形式を増やす場合はリストに追加
    )

    if uploaded_file is not None:
        # アップロードされたファイルの詳細を表示 (デバッグ用)
        # st.write("ファイル詳細:")
        # file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # st.write(file_details)

        # 音声プレイヤーを表示
        st.audio(uploaded_file, format=uploaded_file.type) # type を format に渡す

        # 文字起こしボタン
        if st.button("文字起こしを実行"):
            with st.spinner('音声ファイルを処理し、文字起こしを実行中です...'):
                try:
                    # アップロードされたファイルの内容を読み込む
                    # Streamlit < 1.12 では BytesIO を使う必要があったが、
                    # 最近のバージョンでは read() で bytes が直接取れることが多い
                    content = uploaded_file.read()

                    # RecognitionAudio オブジェクトを作成
                    audio = speech.RecognitionAudio(content=content)

                    # RecognitionConfig オブジェクトを作成
                    # encoding や sample_rate_hertz は多くの場合自動検出されるが、
                    # 特定のファイル形式 (例: LINEAR16のヘッダなしRAWファイルなど) では明示的な指定が必要
                    config = speech.RecognitionConfig(
                        # encoding=speech.RecognitionConfig.AudioEncoding.MP3, # 必要であれば指定
                        # sample_rate_hertz=16000, # 必要であれば指定
                        language_code="ja-JP",  # 日本語を指定
                        enable_automatic_punctuation=True, # 句読点を自動で付与
                        # model="telephony", # ユースケースに合わせてモデルを選択 (例: 電話音声)
                        # audio_channel_count=2, # ステレオ音声の場合など
                    )

                    # Speech-to-Text API を呼び出して文字起こしを実行
                    response = client.recognize(config=config, audio=audio)

                    # 結果の表示
                    st.subheader("文字起こし結果:")
                    if response.results:
                        for result in response.results:
                            st.write(result.alternatives[0].transcript)
                            # st.write(f"信頼度: {result.alternatives[0].confidence:.2f}") # 信頼度も表示したい場合
                    else:
                        st.warning("音声から文字を認識できませんでした。")

                except Exception as e:
                    st.error(f"文字起こし中にエラーが発生しました: {e}")
                    st.error("考えられる原因:")
                    st.error("- アップロードされたファイル形式がサポートされていない、または破損している。")
                    st.error("- 音声が短すぎる、または無音部分が多い。")
                    st.error("- Google Cloud STT API の制限に達した。")
                    st.error("- 認識設定（エンコーディング等）がファイルと合っていない（通常は自動検出）。")

# ----- エラーハンドリング (認証情報の読み込み失敗など) -----
except KeyError as e:
    st.error(f"Streamlit Secrets の設定エラー: '{e}' が見つかりません。")
    st.error("Streamlit Community Cloud のアプリ設定で、Secrets に `google_credentials_json` というキー名で、サービスアカウントキーの JSON 内容全体が正しく設定されているか確認してください。")
except FileNotFoundError: # ローカル実行で secrets.toml がない場合
     st.error("`.streamlit/secrets.toml` が見つかりません。")
     st.error("ローカルで実行する場合、プロジェクトフォルダに `.streamlit/secrets.toml` を作成し、認証情報を記述してください。")
except json.JSONDecodeError:
    st.error("Streamlit Secrets に設定された認証情報 (JSON) の形式が正しくありません。")
    st.error("サービスアカウントキーの JSON ファイルの内容が正しくコピーされているか確認してください。")
except Exception as e:
    st.error(f"認証情報の読み込みまたはクライアント初期化中に予期せぬエラーが発生しました: {e}")
    st.error("Secrets の内容や形式、Google Cloud の認証設定を確認してください。")