import streamlit as st
import pandas as pd
import speech_recognition as sr
import openai
import threading
import time
import os

# OpenAI API 키를 Streamlit Secrets에서 가져오기
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=api_key)
except KeyError:
    st.error("⚠️ OPENAI_API_KEY가 Streamlit Secrets에 설정되지 않았습니다!")
    st.info("Streamlit Cloud에서 App settings > Secrets에 다음과 같이 추가해주세요:")
    st.code('OPENAI_API_KEY = "your-api-key-here"', language="toml")
    st.stop()
except Exception as e:
    st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    st.stop()

# 전역 변수로 녹음 상태 관리
recording_audio = None
stop_recording = False

def recognize_speech_with_interrupt():
    """자동 종료 + 수동 종료 가능한 음성 인식 (PC용)"""
    global recording_audio, stop_recording
    recording_audio = None  # 초기화
    stop_recording = False  # 초기화
    
    try:
        recognizer = sr.Recognizer()
        
        # 음성 인식 설정 조정 (말 끝남 감지 개선)
        recognizer.pause_threshold = 1.5  # 1.5초 정도 멈추면 종료
        recognizer.energy_threshold = 300  # 소음 임계값 조정
        recognizer.non_speaking_duration = 0.8  # 말하지 않는 시간 조정 (더 짧게)
        
        def listen_in_background():
            global recording_audio, stop_recording
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    try:
                        # 자동 종료 모드로 녹음 (말 끝남 감지 개선)
                        recording_audio = recognizer.listen(source, timeout=3, phrase_time_limit=30)
                    except sr.WaitTimeoutError:
                        # 타임아웃 발생 시 수동 종료 모드로 전환
                        try:
                            recording_audio = recognizer.listen(source, timeout=30, phrase_time_limit=60)
                        except Exception as e:
                            pass
            except Exception as e:
                pass
        
        # 백그라운드에서 녹음 시작
        listen_thread = threading.Thread(target=listen_in_background)
        listen_thread.daemon = True
        listen_thread.start()
        
        # 녹음 완료 대기 (non-blocking으로 변경)
        max_wait_time = 35  # 최대 대기 시간 (초)
        wait_start = time.time()
        
        while listen_thread.is_alive() and not stop_recording and (time.time() - wait_start < max_wait_time):
            time.sleep(0.1)  # 짧은 간격으로 체크
        
        if listen_thread.is_alive():
            # 스레드가 아직 실행 중이면 강제 종료 시그널
            stop_recording = True
            listen_thread.join(timeout=1)
        
        if recording_audio and not stop_recording:
            try:
                text = recognizer.recognize_google(recording_audio, language='ko-KR')
                return text
            except sr.UnknownValueError:
                return "음성을 인식할 수 없습니다."
            except sr.RequestError as e:
                return f"Google Speech Recognition 서비스에 접근할 수 없습니다: {e}"
        else:
            return "녹음이 중단되었습니다."
    except Exception as e:
        return f"마이크 접근 실패: {e}. 웹 브라우저에서는 하단의 웹 음성 인식을 사용하세요."

def transcribe_audio_with_whisper(audio_bytes):
    """OpenAI Whisper를 사용하여 오디오를 텍스트로 변환 (iPad 호환)"""
    try:
        # 임시 파일로 저장
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Whisper API로 전사
        with open("temp_audio.wav", "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
        
        # 임시 파일 삭제
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        
        return response.text
    except Exception as e:
        st.error(f"음성 인식 실패: {e}")
        # 임시 파일 정리
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        return None

def correct_transcription_with_prompt(user_input, system_prompt, user_prompt):
    """프롬프트를 사용하여 텍스트 교정"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        st.error(f"프롬프트 처리 실패: {e}")
        return None

def apply_tm_corrections(text, tm_df):
    """TM 데이터를 활용하여 텍스트 교정"""
    if tm_df is None or tm_df.empty:
        return text
    
    corrected_text = text
    
    # TM 데이터의 각 행을 순회하며 교정 적용
    for idx, row in tm_df.iterrows():
        # 컬럼명이 다를 수 있으므로 첫 번째와 두 번째 컬럼 사용
        if len(row) >= 2:
            source_text = str(row.iloc[0]).strip()  # 원본 텍스트
            target_text = str(row.iloc[1]).strip()  # 교정된 텍스트
            
            # 빈 값이 아닌 경우에만 교정 적용
            if source_text and target_text and source_text != 'nan' and target_text != 'nan':
                corrected_text = corrected_text.replace(source_text, target_text)
    
    return corrected_text

def translate_to_english(text):
    """검수된 텍스트를 영어로 번역"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translator. Translate Korean text to English. Return ONLY the English translation, no explanations, no quotes, no additional text."},
                {"role": "user", "content": f"Translate the following text to English: {text}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        st.error(f"번역 처리 실패: {e}")
        return None

def process_text_input(user_input, input_type="음성"):
    """텍스트 입력을 처리하는 공통 함수"""
    if not user_input:
        return
    
    # 입력 시간 기록
    input_completed_time = time.strftime("%H:%M:%S", time.localtime())
    
    st.session_state.recognized_text = user_input
    
    # 1단계: LLM 교정 적용 (원본 텍스트 사용)
    user_prompt = st.session_state.saved_user_prompt_template.replace("{transcription}", user_input)
    
    corrected_text = correct_transcription_with_prompt(user_input, st.session_state.saved_system_prompt, user_prompt)
    correction_completed_time = time.strftime("%H:%M:%S", time.localtime())
    st.session_state.corrected_text = corrected_text
    
    # 2단계: TM 교정 적용 (검수된 텍스트 사용)
    tm_corrected_text = apply_tm_corrections(corrected_text, st.session_state.get('tm_df'))
    tm_completed_time = time.strftime("%H:%M:%S", time.localtime())
    st.session_state.tm_corrected_text = tm_corrected_text
    
    # 3단계: 번역 (TM 교정된 텍스트 사용)
    if tm_corrected_text:
        translated_text = translate_to_english(tm_corrected_text)
        translation_completed_time = time.strftime("%H:%M:%S", time.localtime())
        
        if translated_text:
            st.session_state.translated_text = translated_text
    
    # 디버깅 정보를 세션 상태에 저장
    debug_info = {
        "처리 완료 시간": f"""📝 {input_type} 입력 완료 시간: {input_completed_time}
🔍 검수 LLM 처리 완료 시간: {correction_completed_time}
📊 TM 교정 완료 시간: {tm_completed_time}""",
        "System Prompt": st.session_state.saved_system_prompt,
        "User Prompt": user_prompt
    }
    
    # 번역 시간 추가 (있는 경우)
    if 'translation_completed_time' in locals():
        debug_info["처리 완료 시간"] += f"\n🌐 번역 LLM 처리 완료 시간: {translation_completed_time}"
    
    # TM 정보 추가
    if st.session_state.get('tm_df') is not None:
        tm_status = "✅ TM 교정 적용됨" if corrected_text != tm_corrected_text else "➖ TM 교정 변경사항 없음"
        debug_info["TM 정보"] = f"📊 TM 항목 수: {len(st.session_state.tm_df)}개\n{tm_status}"
    
    st.session_state.debug_info = debug_info

@st.dialog("System Prompt", width="large")
def show_system_prompt():
    st.markdown("### 🤖 System Prompt")
    
    # 큰 텍스트 영역으로 표시 (세로 스크롤 가능)
    st.text_area(
        "프롬프트 내용",
        value=st.session_state.saved_system_prompt,
        height=400,
        disabled=True,
        label_visibility="collapsed"
    )

@st.dialog("User Prompt Template", width="large")
def show_user_prompt():
    st.markdown("### 👤 User Prompt Template")
    
    # 큰 텍스트 영역으로 표시 (세로 스크롤 가능)
    st.text_area(
        "프롬프트 내용",
        value=st.session_state.saved_user_prompt_template,
        height=400,
        disabled=True,
        label_visibility="collapsed"
    )

@st.dialog("System Prompt 편집", width="large")
def edit_system_prompt():
    st.markdown("### 🤖 System Prompt 편집")
    st.markdown("큰 화면에서 편집한 후 저장하면 메인 편집창에 바로 반영됩니다.")
    
    # 편집 가능한 텍스트 영역
    edited_prompt = st.text_area(
        "프롬프트를 편집하세요",
        value=st.session_state.saved_system_prompt,
        height=400,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 저장하고 닫기", use_container_width=True):
            st.session_state.saved_system_prompt = edited_prompt
            st.success("✅ System Prompt가 저장되었습니다!")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("❌ 취소", use_container_width=True):
            st.rerun()

@st.dialog("User Prompt Template 편집", width="large")
def edit_user_prompt():
    st.markdown("### 👤 User Prompt Template 편집")
    st.markdown("큰 화면에서 편집한 후 저장하면 메인 편집창에 바로 반영됩니다.")
    
    # 편집 가능한 텍스트 영역
    edited_prompt = st.text_area(
        "프롬프트를 편집하세요",
        value=st.session_state.saved_user_prompt_template,
        height=400,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 저장하고 닫기", use_container_width=True):
            st.session_state.saved_user_prompt_template = edited_prompt
            st.success("✅ User Prompt Template이 저장되었습니다!")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("❌ 취소", use_container_width=True):
            st.rerun()

def main():
    st.set_page_config(
        page_title="STT 교정 테스트",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 STT 교정 테스트")
    st.markdown("**iPad 및 웹 환경 완벽 지원! 🚀**")

    # 사이드바에 탭 기능 추가
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # 탭 생성
        tab1, tab2 = st.tabs(["📝 프롬프트", "📊 TM"])
        
        # 세션 상태 초기화
        if 'saved_system_prompt' not in st.session_state:
            st.session_state.saved_system_prompt = "You are a meticulous proofreader for the Incheon Main Customs Office. Your task is to correct spelling and transcription errors in Korean text. Return ONLY the corrected Korean text without any explanations, comments, or additional text."
        if 'saved_user_prompt_template' not in st.session_state:
            st.session_state.saved_user_prompt_template = "Please correct any spelling or transcription errors in this Korean text: {transcription}"
        
        # 프롬프트 설정 탭
        with tab1:
            # System Prompt 편집
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("#### 🤖 System Prompt")
            with col2:
                if st.button("🔍", key="edit_system", help="큰 화면에서 편집하기"):
                    edit_system_prompt()
            
            system_prompt_input = st.text_area("", 
                                             value=st.session_state.saved_system_prompt,
                                             height=120,
                                             key="system_prompt_input",
                                             label_visibility="collapsed")
            
            st.markdown("---")
            
            # User Prompt Template 편집
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("#### 👤 User Prompt Template")
            with col2:
                if st.button("🔍", key="edit_user", help="큰 화면에서 편집하기"):
                    edit_user_prompt()
            
            user_prompt_template_input = st.text_area("", 
                                                    value=st.session_state.saved_user_prompt_template,
                                                    height=80,
                                                    key="user_prompt_input",
                                                    label_visibility="collapsed")
            
            # 버튼 섹션
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 저장", key="save_prompt", use_container_width=True):
                    st.session_state.saved_system_prompt = system_prompt_input
                    st.session_state.saved_user_prompt_template = user_prompt_template_input
                    st.success("✅ 저장됨")
            
            with col2:
                if st.button("🔄 초기화", key="reset_prompt", use_container_width=True):
                    st.session_state.saved_system_prompt = "You are a **meticulous proofreader** working for the **{{주제}}**.\n\n## ROLE\nYour task is to correct transcription errors in text produced by a speech-to-text (STT) system. Your most important duty is to detect and correct misrecognized words related to {{주제}}, including both proper nouns and common nouns.\n\n## CORRECTION RULES\n- Correct spelling, spacing, capitalization, and punctuation errors.\n- Always produce corrections in **the same language as the original input**. For example:\n    - If the text is in Korean, correct it in Korean.\n    - If the text is in English, correct it in English.\n    - If the text is in Chinese, correct it in Chinese.\n- For all words, including proper nouns and general vocabulary, fix typos or misrecognized words.\n- For proper nouns, perform fuzzy matching:\n    - If a transcription contains a word similar in spelling or pronunciation to any proper noun in the list below, replace it with the correct spelling, converted to the script or phonetic transcription used in the output language.\n\n- For Korean proper nouns:\n    - Always correct proper nouns to the standard spelling, then transcribe them using the script or phonetic convention typically used in the output language for foreign names, unless there is an official or widely accepted translation.\n    - Never leave proper nouns in Hangul in non-Korean texts.\n    - Examples:\n        - Use Latin letters (romanization) in English, Spanish, French, German, Italian, Portuguese, Indonesian, Dutch, Finnish, Croatian, Czech, Slovak, Polish, Hungarian, Swedish, Malay, Turkish, Tagalog, Swahili, Uzbek.\n        - Use Katakana in Japanese (e.g. ハンサンド).\n        - Use Hanzi (Chinese characters) or pinyin in Chinese (Simplified, Traditional, Cantonese) if widely accepted.\n        - Use local phonetic script in languages such as Thai, Arabic, Russian, Greek, Hebrew, Hindi, Mongolian, Persian, Ukrainian.\n        - Use Hangul in Korean.\n- Do NOT answer any questions.\n- Do NOT explain corrections.\n- Do NOT rephrase or simplify sentences.\n- Only perform necessary corrections as defined above.\n\n## PROPER NOUN LIST (STANDARD FORMS ONLY)\n{{고유단어리스트}}"
                    st.session_state.saved_user_prompt_template = "You are a meticulous proofreader for {{주제}}.\n\n## TASK\nYour only task is to correct spelling, transcription, spacing, punctuation, or typographical errors in the given text.\n\n- The input text may contain Korean, English, Chinese, Japanese, or other languages, or a mixture of them.\n- Keep the text in its original language. Do NOT translate the entire text into another language.\n- However, for Korean proper nouns:\n    - Correct them to their official spelling from the provided proper noun list.\n    - Then transcribe them using the writing system or phonetic convention typically used in the output language for foreign names, unless there is an official or widely accepted translation.\n    - Never leave proper nouns in Hangul in non-Korean texts.\n- For all other words, correct only obvious spelling or transcription mistakes.\n- Do NOT answer questions or explain corrections.\n- Do NOT paraphrase or simplify sentences.\n\n## Origin Transcription:\n{transcription}\n\n## Corrected Transcription:"
                    st.rerun()
            
            # 현재 프롬프트 미리보기
            st.markdown("#### 📋 현재 프롬프트")
            
            # System Prompt 
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("**System**")
                st.text(st.session_state.saved_system_prompt[:50] + "..." if len(st.session_state.saved_system_prompt) > 50 else st.session_state.saved_system_prompt)
            
            with col2:
                if st.button("🔍", key="show_system", help="System Prompt 전체보기"):
                    show_system_prompt()
            
            st.markdown("---")
            
            # User Prompt Template
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("**User**")
                st.text(st.session_state.saved_user_prompt_template[:50] + "..." if len(st.session_state.saved_user_prompt_template) > 50 else st.session_state.saved_user_prompt_template)
            
            with col2:
                if st.button("🔍", key="show_user", help="User Prompt 전체보기"):
                    show_user_prompt()
        
        # TM 설정 탭
        with tab2:
            st.markdown("#### 📊 TM")
            
            # TM 파일 업로드
            uploaded_tm_file = st.file_uploader(
                "TM 파일 업로드", 
                type=['xlsx', 'csv'],
                help="번역 메모리 파일을 업로드하세요. 첫 번째 컬럼은 원본 텍스트, 두 번째 컬럼은 교정된 텍스트여야 합니다."
            )
            
            # TM 데이터 처리
            if uploaded_tm_file is not None:
                try:
                    if uploaded_tm_file.name.endswith('.xlsx'):
                        tm_df = pd.read_excel(uploaded_tm_file, dtype=str)
                    else:
                        tm_df = pd.read_csv(uploaded_tm_file, dtype=str)
                    
                    # 세션 상태에 TM 데이터 저장
                    st.session_state.tm_df = tm_df
                    
                    st.success(f"✅ TM 파일 로드 완료! ({len(tm_df)}개 항목)")
                    
                    # TM 데이터 미리보기
                    with st.expander("TM 데이터 미리보기"):
                        st.dataframe(tm_df.head(10))
                        
                except Exception as e:
                    st.error(f"TM 파일 로드 실패: {e}")
                    st.session_state.tm_df = None
            else:
                # TM 파일이 없으면 세션 상태 초기화
                if 'tm_df' not in st.session_state:
                    st.session_state.tm_df = None
            
            # 현재 TM 상태 표시
            if st.session_state.get('tm_df') is not None:
                st.info(f"🔄 현재 TM: {len(st.session_state.tm_df)}개 항목 활성화됨")
                
                # TM 관리 버튼들
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ TM 삭제", key="clear_tm", use_container_width=True):
                        st.session_state.tm_df = None
                        st.success("TM 데이터가 삭제되었습니다!")
                        st.rerun()
                
                with col2:
                    if st.button("📊 TM 통계", key="tm_stats", use_container_width=True):
                        with st.expander("TM 통계 정보", expanded=True):
                            st.write(f"**총 항목 수:** {len(st.session_state.tm_df)}")
                            st.write(f"**컬럼 수:** {len(st.session_state.tm_df.columns)}")
                            st.write(f"**컬럼명:** {', '.join(st.session_state.tm_df.columns.tolist())}")
            else:
                st.info("📝 TM 파일이 업로드되지 않았습니다")
                st.markdown("---")
                st.markdown("**TM 파일 형식 안내:**")
                st.markdown("- Excel (.xlsx) 또는 CSV 파일")

    # 음성 및 텍스트 입력
    st.subheader("🎯 음성 및 텍스트 입력")
    
    # 세션 상태 초기화
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = None
    if 'tm_corrected_text' not in st.session_state:
        st.session_state.tm_corrected_text = None
    if 'corrected_text' not in st.session_state:
        st.session_state.corrected_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None

    # 음성 입력 부분
    st.markdown("#### 🎤 음성으로 입력하기")
    st.markdown("**iPad Safari 웹 음성 인식 🚀**")
    
    # Web Speech API 컴포넌트
    speech_html = """
    <div style="padding: 20px; border: 2px solid #1f77b4; border-radius: 10px; background-color: #f0f8ff; margin: 10px 0;">
        <h4 style="margin-top: 0; color: #1f77b4;">🎤 실시간 웹 음성 인식</h4>
        <p style="margin: 5px 0; color: #666;">iPad Safari, Chrome, Edge 등에서 지원</p>
        
        <button id="startBtn" onclick="startRecognition()" 
            style="background-color: #1f77b4; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 8px 4px; font-size: 16px;">
            🎤 마이크 시작
        </button>
        <button id="stopBtn" onclick="stopRecognition()" 
            style="background-color: #dc3545; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 8px 4px; font-size: 16px;" disabled>
            🔴 녹음 중지
        </button>
        
        <div id="status" style="margin: 15px 0; font-weight: bold; color: #28a745;">
            🟢 준비됨 - 마이크 권한 허용 후 시작하세요
        </div>
        
        <div id="result" style="margin: 15px 0; padding: 15px; border: 2px solid #e9ecef; border-radius: 8px; min-height: 80px; background-color: white; font-family: 'Noto Sans KR', sans-serif;">
            <em style="color: #6c757d;">인식된 텍스트가 여기에 실시간으로 표시됩니다...</em>
        </div>
        
        <div style="margin-top: 15px;">
            <button id="processBtn" onclick="processRecognizedText()" 
                style="background-color: #28a745; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; margin: 4px; font-size: 16px;" disabled>
                ✅ 인식된 텍스트 처리하기
            </button>
            <button id="clearBtn" onclick="clearResult()" 
                style="background-color: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; margin: 4px;">
                🗑️ 지우기
            </button>
        </div>
    </div>

    <script>
    let recognition;
    let isRecognizing = false;
    let finalTranscript = '';

    function startRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'ko-KR';
            
            recognition.onstart = function() {
                isRecognizing = true;
                updateStatus('🎤 음성 인식 중... 말씀하세요! (1.5초 멈추면 자동 종료)', '#dc3545');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('processBtn').disabled = true;
                finalTranscript = '';
            };
            
            recognition.onresult = function(event) {
                let interimTranscript = '';
                let newFinalTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        newFinalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }
                
                finalTranscript += newFinalTranscript;
                
                let displayText = '';
                if (finalTranscript) {
                    displayText += '<div style="color: #000; font-weight: bold; margin-bottom: 8px;">✅ 확정: ' + finalTranscript + '</div>';
                }
                if (interimTranscript) {
                    displayText += '<div style="color: #666; font-style: italic;">⏳ 인식 중: ' + interimTranscript + '</div>';
                }
                
                document.getElementById('result').innerHTML = displayText || '<em style="color: #6c757d;">음성을 듣고 있습니다...</em>';
                
                if (finalTranscript) {
                    document.getElementById('processBtn').disabled = false;
                }
            };
            
            recognition.onerror = function(event) {
                let errorMsg = '❌ 오류: ' + event.error;
                if (event.error === 'not-allowed') {
                    errorMsg += ' (마이크 권한을 허용해주세요)';
                } else if (event.error === 'no-speech') {
                    errorMsg += ' (음성이 감지되지 않습니다)';
                }
                updateStatus(errorMsg, '#dc3545');
                stopRecognition();
            };
            
            recognition.onend = function() {
                if (isRecognizing) {
                    let msg = finalTranscript ? 
                        '✅ 음성 인식 완료! "인식된 텍스트 처리하기" 버튼을 눌러주세요.' : 
                        '⏹️ 음성 인식이 종료되었습니다.';
                    updateStatus(msg, '#28a745');
                }
                isRecognizing = false;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            };
            
            recognition.start();
        } else {
            updateStatus('❌ 이 브라우저는 음성 인식을 지원하지 않습니다. Chrome 또는 Safari를 사용해주세요.', '#dc3545');
        }
    }

    function stopRecognition() {
        if (recognition && isRecognizing) {
            recognition.stop();
        }
        isRecognizing = false;
    }

    function updateStatus(message, color) {
        const statusEl = document.getElementById('status');
        statusEl.innerHTML = message;
        statusEl.style.color = color;
    }

    function processRecognizedText() {
        if (finalTranscript) {
            // Streamlit에 텍스트 전달을 위해 hidden input 사용
            const hiddenInput = document.createElement('input');
            hiddenInput.type = 'hidden';
            hiddenInput.id = 'recognizedText';
            hiddenInput.value = finalTranscript;
            document.body.appendChild(hiddenInput);
            
            // 사용자에게 복사 안내
            navigator.clipboard.writeText(finalTranscript).then(function() {
                updateStatus('📋 텍스트가 클립보드에 복사되었습니다! 아래 입력창에 붙여넣고 처리하기를 눌러주세요.', '#17a2b8');
            }).catch(function() {
                updateStatus('📝 인식된 텍스트: "' + finalTranscript + '" - 아래 입력창에 복사해서 붙여넣어 주세요.', '#17a2b8');
            });
        }
    }

    function clearResult() {
        finalTranscript = '';
        document.getElementById('result').innerHTML = '<em style="color: #6c757d;">인식된 텍스트가 여기에 실시간으로 표시됩니다...</em>';
        document.getElementById('processBtn').disabled = true;
        updateStatus('🟢 준비됨 - 마이크 권한 허용 후 시작하세요', '#28a745');
    }
    </script>
    """
    
    # HTML 컴포넌트 표시
    st.components.v1.html(speech_html, height=350)
    
    # 음성 인식 결과 처리용 텍스트 입력
    st.markdown("**인식된 텍스트 처리:**")
    speech_result = st.text_area(
        "위에서 인식된 텍스트를 복사해서 붙여넣으세요:",
        placeholder="음성 인식 후 '✅ 인식된 텍스트 처리하기' 버튼을 누르고 여기에 붙여넣으세요",
        key="speech_input",
        height=100
    )
    
    if speech_result.strip():
        if st.button("✅ 음성 인식 텍스트 처리", key="process_web_speech", use_container_width=True):
            process_text_input(speech_result.strip(), "음성(웹)")
            st.rerun()
    
    # 텍스트 입력 부분 (음성 입력 아래에 추가)
    st.markdown("#### ✏️ 또는 텍스트로 직접 입력하기")
    
    # 텍스트 입력 필드
    text_input = st.text_area("텍스트를 입력하세요:", 
                               height=100,
                               placeholder="예: 안녕하세요. 처리하기를 눌러주세요.")
    
    # 처리하기 버튼
    if st.button("🔄 처리하기", key="text_input_button", use_container_width=True):
        if text_input.strip():
            process_text_input(text_input.strip(), "텍스트")
            st.rerun()
        else:
            st.warning("텍스트를 입력해주세요!")

    # 디버깅 정보 표시 (처리하기 버튼 바로 아래)
    if st.session_state.get('debug_info'):
        with st.expander("🔍 디버깅 정보"):
            for key, value in st.session_state.debug_info.items():
                st.write(f"**{key}:**")
                if key in ["System Prompt", "User Prompt"]:
                    st.code(value, language="text")
                else:
                    st.write(value)

    # 결과 표시
    if st.session_state.get('recognized_text'):
        st.markdown("---")
        st.subheader("📋 처리 결과")
        
        # 결과를 카드 형태로 표시
        with st.container():
            st.markdown("**🔤 입력받은 내용:**")
            st.info(st.session_state.recognized_text)
            # 입력 방식 표시
            if "웹" in str(st.session_state.get('debug_info', {}).get('처리 완료 시간', '')):
                st.caption("🎤 웹 음성 인식으로 입력됨 (iPad 호환)")
            else:
                st.caption("✏️ 직접 텍스트 입력")
                
        if st.session_state.get('corrected_text'):
            with st.container():
                st.markdown("**🔍 검수:**")
                st.success(st.session_state.corrected_text)
            
        if st.session_state.get('tm_corrected_text') and st.session_state.corrected_text != st.session_state.tm_corrected_text:
            with st.container():
                st.markdown("**📊 TM 교정:**")
                st.success(st.session_state.tm_corrected_text)
                
        if st.session_state.get('translated_text'):
            with st.container():
                st.markdown("**🌐 번역:**")
                st.success(st.session_state.translated_text)

if __name__ == "__main__":
    main()
