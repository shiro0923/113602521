()
config.read('config.ini')

# 設定 Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    google_api_key=config["Gemini"]["API_KEY"],
    convert_system_message_to_human=True,
)


speech_config = speechsdk.SpeechConfig(subscription=config['AzureSpeech']['SPEECH_KEY'], 
                                       region=config['AzureSpeech']['SPEECH_REGION'])
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
UPLOAD_FOLDER = 'static'

# 初始化 Azure 翻譯和視覺服務的客戶端
text_translator = TextTranslationClient(
    credential=AzureKeyCredential(config["AzureTranslator"]["Key"]),
    endpoint=config["AzureTranslator"]["EndPoint"],
    region=config["AzureTranslator"]["Region"]# Azure 翻譯服務的區域
)
vision_client = ComputerVisionClient(
    endpoint=config["AzureComputerVision"]["EndPoint"],
    credentials=CognitiveServicesCredentials(config["AzureComputerVision"]["Key"])
)

# 初始化 Flask 應用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import sys
import configparser
import logging
import requests
import librosa
import os
import uuid
import time

from pydub import AudioSegment

import azure.cognitiveservices.speech as speechsdk
from io import BytesIO
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, AudioMessage

#from configparser import ConfigParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# 設定檔案保存時間（秒），例如 24 小時
MAX_FILE_AGE = 12 * 60 * 60  
MAX_FILES = 40  # 最大檔案數量
# 設定日誌
logging.basicConfig(level=logging.INFO)





# 讀取設定檔案
config = configparser.ConfigParser
# 初始化 LINE Bot API
channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')

handler = WebhookHandler(channel_secret)
configuration = Configuration(
    access_token=channel_access_token
)



# 定義 LINE Webhook 入口點
@app.route("/callback", methods=['POST'])
def callback():
    # 從 HTTP Header 取得 X-Line-Signature
    signature = request.headers['X-Line-Signature']
    # 取得請求的主體內容
    body = request.get_data(as_text=True)
    
    # 格式化輸出
    formatted_body = "\n".join(body.split(","))  # 以逗號為分隔符進行換行（或其他分隔符）
    app.logger.info("Formatted Request body: \n" + formatted_body)

    # 解析 Webhook 資料並驗證簽名
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # 驗證失敗回傳 400 錯誤
    return 'OK'


# 處理文字訊息事件
@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    # 呼叫 Azure 翻譯服務，翻譯使者傳來的文字
    translation_result, r_t_0, r_t_1, d_l_0, d_l_1 = azure_translate(event.message.text)
    print(translation_result)

    audio_duration_0, file_name_0 = azure_speech(r_t_0, d_l_0)
    audio_duration_1, file_name_1 = azure_speech(r_t_1, d_l_1)
    print(file_name_0)

    # 使用 LINE Messaging API 回覆翻譯結果
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,  # 回覆用的 token
                messages = [TextMessage(text=translation_result),
                            AudioMessage(
                                originalContentUrl=file_name_0,
                                duration=audio_duration_0
                            ),
                            AudioMessage(
                                originalContentUrl=file_name_1,
                                duration=audio_duration_1
                            )
                ]  # 翻譯結果作為回應訊息
            )
        )



@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    content_url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
    headers = {"Authorization": f"Bearer {channel_access_token}"}

    response = requests.get(content_url, headers=headers, stream=True)
    if response.status_code != 200:
        print(f"無法獲取圖片內容：{response.status_code}")
        return

    image_stream = BytesIO(response.content)
    #Azure 圖片分析
    analysis_result, description= analyze_image_with_azure(image_stream)
    # 使用 Google Gemini
    gemini_explain = gemini(description)
    #翻譯&轉語音
    audio_duration, file_name = azure_speech(gemini_explain, "zh-Hant")

    analysis_result = f"{analysis_result}\n解釋：{gemini_explain}"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=analysis_result),
                          AudioMessage(
                                originalContentUrl=file_name,
                                duration=audio_duration
                            )
                ]
            )
        )




def convert_wav_to_mp3(input_path, output_path):
    try:
        # 讀取 WAV 檔案
        audio = AudioSegment.from_wav(input_path)
        
        # 儲存為 MP3 格式
        audio.export(output_path, format="mp3")
        print(f"已將音檔轉換為 MP3 格式並儲存到 {output_path}")
        return output_path
    except Exception as e:
        error_messa = f"轉換 WAV 為 MP3 失敗：{e}"
        response = log_and_return_error(error_messa)
        print(response)
        raise



def clean_old_files(folder_path):
    now = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > MAX_FILE_AGE:
                os.remove(file_path)
                print(f"已刪除過期檔案：{file_path}")



def limit_files_in_folder(folder_path, max_files):
    files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    if len(files) > max_files:
        files.sort(key=os.path.getmtime)  # 按檔案修改時間排序
        for file_to_remove in files[:-max_files]:
            os.remove(file_to_remove)
            print(f"已刪除多餘檔案：{file_to_remove}")



def azure_speech(user_input, detected_l):
    # 在產生新音檔前執行清理
    clean_old_files(app.config['UPLOAD_FOLDER'])
    # 在產生新音檔前執行
    limit_files_in_folder(app.config['UPLOAD_FOLDER'], MAX_FILES)
    # The language of the voice that speaks.
    accent_map = {
        "en": "en-GB-OllieMultilingualNeural",
        "ja": "ja-JP-NanamiNeural",
        "zh-Hans": "zh-CN-XiaohanNeural",
        "zh-Hant": "zh-TW-HsiaoChenNeural"
    }
    speech_config.speech_synthesis_voice_name = accent_map.get(detected_l)
    file_name = f"outputaudio_{uuid.uuid4().hex}.wav"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file_config = speechsdk.audio.AudioOutputConfig(filename=file_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=file_config
    )
    # Receives a text from console input and synthesizes it to wave file.
    result = speech_synthesizer.speak_text_async(user_input).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(
            "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                user_input, file_name
            )
        )
         # 將 WAV 轉換為 MP3
        mp3_name = file_name.replace(".wav", ".mp3")
        mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_name)
        convert_wav_to_mp3(file_path, mp3_path)
        
        
        mp3_url = f"{config['Deploy']['URL']}/static/{os.path.basename(mp3_name)}"

        # 計算 MP3 音檔時長（基於原始 WAV）
        audio_duration = round(librosa.get_duration(path=file_path) * 1000)
        return audio_duration, mp3_url
    
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        error_message_0 = f"Speech synthesis canceled: {cancellation_details.reason}"
        log_and_return_error(error_message_0)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_message_1 = f"Error details: {cancellation_details.error_details}"
            response = log_and_return_error(error_message_1)
            print(response)
        raise



def azure_translate(user_input):
    try:
        target_languages = ["en", "ja", "zh-Hant"]
        response = text_translator.translate(
            body=[user_input], to_language=target_languages
        )

        if not response:
            return "無法取得翻譯結果，請稍後再試。"

        translations = response[0].translations
        detected_language = response[0].detected_language.language

        language_map = {
            "zh-Hant": [translations[1].text, translations[0].text],
            "zh-Hans": [translations[1].text, translations[0].text],
            "ja": [translations[2].text, translations[0].text],
            "en": [translations[2].text, translations[1].text]
        }

        detected_language_map = {
            "zh-Hant": ["ja", "en"],
            "zh-Hans": ["ja", "en"],
            "ja": ["zh-Hant", "en"],
            "en": ["zh-Hant", "ja"]
        }
        
        result_text_0, result_text_1 = language_map.get(detected_language, ["翻譯結果不可用", "翻譯結果不可用"])
        detected_language_0, detected_language_1 = detected_language_map.get(detected_language, ["None", "None"])
        result = "\n".join(language_map.get(detected_language, ["翻譯結果不可用"]))
        return result, result_text_0, result_text_1, detected_language_0, detected_language_1
    except HttpResponseError as exception:
        error_message = f"Azure 翻譯服務錯誤：{exception.error.message}"
        response_e = log_and_return_error(error_message)
        print(response_e)
        raise



def analyze_image_with_azure(image_stream):
    try:
        analysis = vision_client.analyze_image_in_stream(
            image_stream,
            visual_features=["Description", "Objects"]
        )

        description = analysis.description.captions[0].text if analysis.description.captions else "無法辨識圖片內容"
        objects = [obj.object_property for obj in analysis.objects]

        reslut = f"圖片分析結果：\n描述：{description}\n物件：{', '.join(objects)}"
        return reslut, description
    except Exception as e:
        error_message = f"Azure 電腦視覺錯誤：{e}"
        response = log_and_return_error(error_message)
        print(response)
        raise



def gemini(description):
    try:
        question = "describe a scene of" + description + "to a viually repaired within 50 words"
        role_description = "你是一位5歲小孩，請使用繁體中文回答。"
        # 使用 Google Gemini
        messages = [HumanMessage(content=f"{role_description}\n{question}")]
        result = llm.invoke(messages)
        answer = result.content
        return answer
    except:
        error_message = f"Gemini 分析錯誤"
        response = log_and_return_error(error_message)
        print(response)
        raise



def log_and_return_error(message):
    logging.error(message)
    return "操作失敗，請稍後再試。"



if __name__ == "__main__":
    app.run()