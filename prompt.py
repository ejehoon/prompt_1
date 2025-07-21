import streamlit as st
import pandas as pd
import speech_recognition as sr
import openai
import threading
import time
import os

# OpenAI API 키 설정 (Streamlit Cloud secrets 사용)
try:
    # Streamlit Cloud에서 secrets 사용
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # 로컬 환경에서는 환경변수 사용
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets에서 OPENAI_API_KEY를 설정해주세요.")
        st.stop()

client = openai.OpenAI(api_key=api_key)

# 전역 변수로 녹음 상태 관리
recording_audio = None
stop_recording = False

def recognize_speech_with_interrupt():
    """자동 종료 + 수동 종료 가능한 음성 인식"""
    global recording_audio, stop_recording
    recording_audio = None  # 초기화
    recognizer = sr.Recognizer()
    
    # 음성 인식 설정 조정 (말 끝남 감지 개선)
    recognizer.pause_threshold = 1.5  # 1.5초 정도 멈추면 종료
    recognizer.energy_threshold = 300  # 소음 임계값 조정
    recognizer.non_speaking_duration = 0.8  # 말하지 않는 시간 조정 (더 짧게)
    
    def listen_in_background():
        global recording_audio
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
    
    # 녹음 중 표시
    progress_placeholder = st.empty()
    progress_placeholder.info("🎤 녹음 중... (1.5초 멈추면 자동 종료)")
    
    # 녹음 완료 대기
    listen_thread.join()
    progress_placeholder.empty()
    
    if recording_audio:
        try:
            text = recognizer.recognize_google(recording_audio, language='ko-KR')
            return text
        except sr.UnknownValueError:
            return "음성을 인식할 수 없습니다."
        except sr.RequestError as e:
            return f"Google Speech Recognition 서비스에 접근할 수 없습니다: {e}"
    else:
        return "녹음된 오디오가 없습니다."


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
        st.write(f"프롬프트 처리 실패: {e}")
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
        st.write(f"번역 처리 실패: {e}")
        return None


def process_text_input(user_input, input_type="음성"):
    """텍스트 입력을 처리하는 공통 함수"""
    if not user_input:
        return
    
    # 입력 시간 기록
    input_completed_time = time.strftime("%H:%M:%S", time.localtime())
    
    st.session_state.recognized_text = user_input
    
    # 1단계: TM 교정 적용
    tm_corrected_text = apply_tm_corrections(user_input, st.session_state.get('tm_df'))
    tm_completed_time = time.strftime("%H:%M:%S", time.localtime())
    st.session_state.tm_corrected_text = tm_corrected_text
    
    # 2단계: LLM 교정 적용 (TM 교정된 텍스트 사용)
    user_prompt = st.session_state.saved_user_prompt_template.replace("{transcription}", tm_corrected_text)
    
    corrected_text = correct_transcription_with_prompt(tm_corrected_text, st.session_state.saved_system_prompt, user_prompt)
    correction_completed_time = time.strftime("%H:%M:%S", time.localtime())
    
    if corrected_text:
        st.session_state.corrected_text = corrected_text
        
        translated_text = translate_to_english(corrected_text)
        translation_completed_time = time.strftime("%H:%M:%S", time.localtime())
        
        if translated_text:
            st.session_state.translated_text = translated_text
    
    # 디버깅: 각 단계 완료 시간
    with st.expander("디버깅 정보"):
        # 더 강력한 CSS로 expander 가로 길이 최대화
        st.markdown("""
        <style>
        /* Expander 전체 가로 길이 확장 */
        .stExpander {
            width: 100% !important;
            max-width: none !important;
        }
        .stExpander > div {
            width: 100% !important;
            max-width: none !important;
        }
        .stExpander .streamlit-expanderContent {
            width: 100% !important;
            max-width: none !important;
        }
        /* 텍스트 에어리어 가로 길이 확장 */
        .stTextArea textarea {
            width: 100% !important;
            max-width: none !important;
            min-width: 100% !important;
        }
        /* 디버깅 전용 스타일 */
        .debug-wide {
            width: 100% !important;
            max-width: 100vw !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.write("**처리 완료 시간:**")
        st.write(f"📝 {input_type} 입력 완료 시간: {input_completed_time}")
        st.write(f"📊 TM 교정 완료 시간: {tm_completed_time}")
        st.write(f"🔍 검수 LLM 처리 완료 시간: {correction_completed_time}")
        if 'translation_completed_time' in locals():
            st.write(f"🌐 번역 LLM 처리 완료 시간: {translation_completed_time}")
        
        st.markdown("---")
        st.write("**프롬프트 정보:**")
        
        # System Prompt를 최대 가로 길이로 표시
        st.write("System Prompt:")
        st.text_area(
            "system_debug", 
            value=st.session_state.saved_system_prompt, 
            height=200, 
            disabled=True, 
            label_visibility="collapsed",
            key="debug_system_prompt"
        )
        
        # User Prompt를 최대 가로 길이로 표시
        st.write("User Prompt:")
        st.text_area(
            "user_debug", 
            value=user_prompt, 
            height=150, 
            disabled=True, 
            label_visibility="collapsed",
            key="debug_user_prompt"
        )
        
        # TM 적용 여부 표시
        if st.session_state.get('tm_df') is not None:
            st.markdown("---")
            st.write("**TM 정보:**")
            st.write(f"📊 TM 항목 수: {len(st.session_state.tm_df)}개")
            if st.session_state.recognized_text != st.session_state.tm_corrected_text:
                st.write("✅ TM 교정 적용됨")
            else:
                st.write("➖ TM 교정 변경사항 없음")


def main():
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
            st.markdown("#### 🤖 System Prompt")
            system_prompt_input = st.text_area("", 
                                             value=st.session_state.saved_system_prompt,
                                             height=120,
                                             key="system_prompt_input",
                                             label_visibility="collapsed")
            
            st.markdown("#### 👤 User Prompt Template")
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
            with st.expander("📋 현재 프롬프트 미리보기"):
                st.markdown("**System Prompt:**")
                st.text(st.session_state.saved_system_prompt[:100] + "..." if len(st.session_state.saved_system_prompt) > 100 else st.session_state.saved_system_prompt)
                
                st.markdown("**User Prompt Template:**")
                st.text(st.session_state.saved_user_prompt_template[:100] + "..." if len(st.session_state.saved_user_prompt_template) > 100 else st.session_state.saved_user_prompt_template)
        
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
        button_text = "🔴 녹음 중... (클릭하여 종료)"
    else:
        button_text = "🎤 마이크 시작"

    if st.button(button_text, key='mic_button'):
        if not st.session_state.is_recording:
            # 녹음 시작
            st.session_state.is_recording = True
            
            user_input = recognize_speech_with_interrupt()
            
            if user_input:
                process_text_input(user_input, "음성")
                
            st.session_state.is_recording = False
        else:
            # 녹음 종료
            st.session_state.is_recording = False
            global stop_recording
            stop_recording = True

    # 텍스트 입력 부분 (음성 입력 아래에 추가)
    st.markdown("#### ✏️ 또는 텍스트로 직접 입력하기")
    
    # 텍스트 입력 필드
    text_input = st.text_area("텍스트를 입력하세요:", 
                               value=st.session_state.recognized_text if st.session_state.recognized_text else "",
                               key="text_input_area",
                               height=100,
                               placeholder="예: 이것은 텍스트 입력 테스트인가요?")
    
    # 버튼 컬럼
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 처리하기", key="text_input_button", use_container_width=True):
            if text_input:
                process_text_input(text_input, "텍스트")
            else:
                st.warning("텍스트를 입력해주세요!")
    
    with col2:
        if st.button("🗑️ 지우기", key="clear_text_button", use_container_width=True):
            st.session_state.recognized_text = None
            st.session_state.tm_corrected_text = None
            st.session_state.corrected_text = None
            st.session_state.translated_text = None
            st.rerun()

    # 결과 표시
    if st.session_state.recognized_text:
        st.markdown("---")
        st.subheader("📋 처리 결과")
        
        # 결과를 카드 형태로 표시
        with st.container():
            st.markdown("**🔤 입력받은 내용:**")
            st.info(st.session_state.recognized_text)
            
        if st.session_state.tm_corrected_text and st.session_state.tm_corrected_text != st.session_state.recognized_text:
            with st.container():
                st.markdown("**📊 TM 교정:**")
                st.success(st.session_state.tm_corrected_text)
                
        if st.session_state.corrected_text:
            with st.container():
                st.markdown("**🔍 검수:**")
                st.success(st.session_state.corrected_text)
                
        if st.session_state.translated_text:
            with st.container():
                st.markdown("**🌐 번역:**")
                st.success(st.session_state.translated_text)
                
        # 추가 액션 버튼들
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 결과 복사", key="copy_result"):
                if st.session_state.translated_text:
                    st.code(st.session_state.translated_text, language="text")
                    st.success("번역 결과가 표시되었습니다!")
                    
        with col2:
            if st.button("🔄 다시 처리", key="reprocess"):
                if st.session_state.recognized_text:
                    process_text_input(st.session_state.recognized_text, "재처리")
                    
        with col3:
            if st.button("🗑️ 전체 지우기", key="clear_all"):
                st.session_state.recognized_text = None
                st.session_state.tm_corrected_text = None
                st.session_state.corrected_text = None
                st.session_state.translated_text = None
                st.success("모든 결과가 삭제되었습니다!")
                st.rerun()


if __name__ == "__main__":
    main()
