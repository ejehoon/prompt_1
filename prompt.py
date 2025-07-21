import streamlit as st
import pandas as pd
import openai
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

# 웹 음성 인식용 (기존 전역 변수는 더 이상 사용하지 않음)

def recognize_speech_with_interrupt():
    """웹 음성 인식 (iPad Safari 호환) - 기존 함수 시그니처 유지"""
    try:
        # 웹 음성 인식 컴포넌트
        speech_html = """
        <div style="text-align: center; padding: 20px; border: 2px solid #1f77b4; border-radius: 10px; background-color: #f0f8ff; margin: 10px 0;">
            <p id="status" style="font-size: 18px; margin-bottom: 15px;"><strong>🎤 마이크 권한을 허용하고 말씀해주세요</strong></p>
            <input type="text" id="speechResult" placeholder="인식된 텍스트가 여기에 나타납니다" 
                   style="width: 80%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; text-align: center;" readonly>
            <br><br>
            <button onclick="copySpeechResult()" id="copyBtn" 
                    style="background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; display: none;">
                📋 결과 복사하기
            </button>
        </div>
        
        <script>
        function startSpeechRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'ko-KR';
                
                document.getElementById('status').innerHTML = '<strong>🎤 음성을 듣고 있습니다... 말씀하세요!</strong>';
                
                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('speechResult').value = transcript;
                    document.getElementById('status').innerHTML = '<strong>✅ 음성 인식 완료!</strong>';
                    document.getElementById('copyBtn').style.display = 'inline-block';
                    
                    // 자동으로 클립보드에 복사
                    navigator.clipboard.writeText(transcript).catch(function() {
                        console.log('클립보드 복사 실패');
                    });
                };
                
                recognition.onerror = function(event) {
                    if (event.error === 'not-allowed') {
                        document.getElementById('status').innerHTML = '<strong>❌ 마이크 권한을 허용해주세요</strong>';
                    } else if (event.error === 'no-speech') {
                        document.getElementById('status').innerHTML = '<strong>❌ 음성이 감지되지 않았습니다</strong>';
                    } else {
                        document.getElementById('status').innerHTML = '<strong>❌ 오류: ' + event.error + '</strong>';
                    }
                };
                
                recognition.start();
            } else {
                document.getElementById('status').innerHTML = '<strong>❌ 이 브라우저는 음성 인식을 지원하지 않습니다</strong>';
            }
        }
        
        function copySpeechResult() {
            const text = document.getElementById('speechResult').value;
            navigator.clipboard.writeText(text).then(function() {
                alert('클립보드에 복사되었습니다! 아래 텍스트 입력창에 붙여넣어 주세요.');
            });
        }
        
        // 자동 시작
        setTimeout(startSpeechRecognition, 500);
        </script>
        """
        
        st.components.v1.html(speech_html, height=200)
        
        # 음성 인식 결과를 받을 텍스트 입력 (기존 플로우 유지용)
        st.info("🎤 위에서 음성 인식이 완료되면 '📋 결과 복사하기' 버튼을 클릭하여 결과를 복사하고, 아래에 붙여넣어 주세요.")
        
        # 사용자 입력 필드
        recognized_text = st.text_input(
            "음성 인식 결과를 붙여넣으세요:",
            placeholder="위에서 인식된 텍스트를 복사해서 여기에 붙여넣으세요",
            key=f"web_speech_input_{int(time.time())}"
        )
        
        if recognized_text and recognized_text.strip():
            return recognized_text.strip()
        else:
            return "음성을 인식할 수 없습니다."
            
    except Exception as e:
        return f"음성 인식 실패: {e}"

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
    
    st.title("STT 교정 테스트")

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
    st.subheader("음성 및 텍스트 입력")
    
    # 세션 상태 초기화
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = None
    if 'tm_corrected_text' not in st.session_state:
        st.session_state.tm_corrected_text = None
    if 'corrected_text' not in st.session_state:
        st.session_state.corrected_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    # 음성 입력 부분
    st.markdown("#### 🎤 음성으로 입력하기")
    
    # 마이크 버튼
    if st.session_state.is_recording:
        button_text = "🔴 녹음 중 (클릭하여 종료)"
        button_type = "secondary"
    else:
        button_text = "🎤 마이크 시작"
        button_type = "primary"

    if st.button(button_text, key='mic_button', type=button_type):
        if not st.session_state.is_recording:
            # 녹음 시작
            st.session_state.is_recording = True
            st.rerun()  # 버튼 상태를 즉시 업데이트
        else:
            # 녹음 종료
            st.session_state.is_recording = False
            st.rerun()  # 버튼 상태를 즉시 업데이트

    # 녹음 중일 때 음성 인식 실행
    if st.session_state.is_recording:
        with st.spinner("🎤 음성을 인식하는 중... (1.5초 멈추면 자동 종료)"):
            user_input = recognize_speech_with_interrupt()
            
        # 녹음 완료 후 처리
        st.session_state.is_recording = False
        
        if user_input and "중단되었습니다" not in user_input and "인식할 수 없습니다" not in user_input and "웹 음성 인식 실패" not in user_input:
            process_text_input(user_input, "음성")
            st.rerun()
        elif user_input:
            if "중단되었습니다" in user_input:
                st.info("🔴 녹음이 중단되었습니다.")
            else:
                st.warning(f"⚠️ {user_input}")
        
        st.rerun()  # 버튼 상태 업데이트

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
