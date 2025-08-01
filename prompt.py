import streamlit as st
import pandas as pd
import speech_recognition as sr
import openai
import threading
import time
import json

# OpenAI API 키를 Streamlit Secrets에서 안전하게 가져오기
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=api_key)
except KeyError:
    st.error("🔑 OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets에 'OPENAI_API_KEY'를 추가해주세요.")
    st.info("💡 **설정 방법:**\n1. Streamlit Cloud 대시보드에서 앱 설정으로 이동\n2. Secrets 탭에서 다음과 같이 추가:\n```\nOPENAI_API_KEY = \"your-api-key-here\"\n```")
    st.stop()
except Exception as e:
    st.error(f"API 키 설정 중 오류 발생: {e}")
    st.stop()

# 전역 변수로 녹음 상태 관리
recording_audio = None
stop_recording = False

def recognize_speech_with_interrupt():
    """자동 종료 + 수동 종료 가능한 음성 인식"""
    global recording_audio, stop_recording
    recording_audio = None  # 초기화
    stop_recording = False  # 초기화
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

@st.dialog("System Prompt (JSON 뷰)", width="large")
def show_system_prompt_json():
    st.markdown("### 🤖 System Prompt - JSON 형식")
    
    try:
        # JSON 파싱 시도
        parsed_json = json.loads(st.session_state.saved_system_prompt)
        st.json(parsed_json)
    except json.JSONDecodeError:
        # JSON이 아닌 경우 일반 텍스트로 표시하되 JSON 형식으로 보이도록 시도
        try:
            # 문자열 내의 이스케이프 문자를 실제 문자로 변환
            formatted_text = st.session_state.saved_system_prompt.replace('\\n', '\n').replace('\\t', '\t')
            st.code(formatted_text, language="json")
        except:
            st.text_area(
                "프롬프트 내용 (JSON 파싱 실패)",
                value=st.session_state.saved_system_prompt,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )

@st.dialog("User Prompt Template (JSON 뷰)", width="large")
def show_user_prompt_json():
    st.markdown("### 👤 User Prompt Template - JSON 형식")
    
    try:
        # JSON 파싱 시도
        parsed_json = json.loads(st.session_state.saved_user_prompt_template)
        st.json(parsed_json)
    except json.JSONDecodeError:
        # JSON이 아닌 경우 일반 텍스트로 표시하되 JSON 형식으로 보이도록 시도
        try:
            # 문자열 내의 이스케이프 문자를 실제 문자로 변환
            formatted_text = st.session_state.saved_user_prompt_template.replace('\\n', '\n').replace('\\t', '\t')
            st.code(formatted_text, language="json")
        except:
            st.text_area(
                "프롬프트 내용 (JSON 파싱 실패)",
                value=st.session_state.saved_user_prompt_template,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )

def main():
    # 모드에 따라 제목 변경
    if st.session_state.get('app_mode', 'stt') == "evaluate":
        st.title("🧪 프롬프트 평가 대시보드")
    else:
        st.title("STT 교정 테스트")

    # 사이드바에 탭 기능 추가
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["📝 프롬프트", "📊 TM", "🧪 평가하기"])
        
        # 세션 상태 초기화
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = "stt"  # "stt" 또는 "evaluate"
        if 'saved_system_prompt' not in st.session_state:
            st.session_state.saved_system_prompt = "test"
        if 'saved_user_prompt_template' not in st.session_state:
            st.session_state.saved_user_prompt_template = "You are a meticulous proofreader for {{주제}}.\n\n## TASK\nYour only task is to correct spelling, transcription, spacing, punctuation, or typographical errors in the given text.\n\n- The input text may contain Korean, English, Chinese, Japanese, or other languages, or a mixture of them.\n- Keep the text in its original language. Do NOT translate the entire text into another language.\n- However, for Korean proper nouns:\n    - Correct them to their official spelling from the provided proper noun list.\n    - Then transcribe them using the writing system or phonetic convention typically used in the output language for foreign names, unless there is an official or widely accepted translation.\n    - Never leave proper nouns in Hangul in non-Korean texts.\n- For all other words, correct only obvious spelling or transcription mistakes.\n- Do NOT answer questions or explain corrections.\n- Do NOT paraphrase or simplify sentences.\n\n## Origin Transcription:\n{transcription}\n\n## Corrected Transcription:"
        
        # 평가 모드 관련 세션 상태 초기화
        if 'test_cases' not in st.session_state:
            st.session_state.test_cases = []
        if 'prompt_variables' not in st.session_state:
            st.session_state.prompt_variables = {}
        if 'evaluation_prompt' not in st.session_state:
            st.session_state.evaluation_prompt = ""
        
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
                if st.button("예시", key="reset_prompt", use_container_width=True):
                    st.session_state.saved_system_prompt = "You are a **meticulous proofreader** working for the **{{주제}}**.\n\n## ROLE\nYour task is to correct transcription errors in text produced by a speech-to-text (STT) system. Your most important duty is to detect and correct misrecognized words related to {{주제}}, including both proper nouns and common nouns.\n\n## CORRECTION RULES\n- Correct spelling, spacing, capitalization, and punctuation errors.\n- Always produce corrections in **the same language as the original input**. For example:\n    - If the text is in Korean, correct it in Korean.\n    - If the text is in English, correct it in English.\n    - If the text is in Chinese, correct it in Chinese.\n- For all words, including proper nouns and general vocabulary, fix typos or misrecognized words.\n- For proper nouns, perform fuzzy matching:\n    - If a transcription contains a word similar in spelling or pronunciation to any proper noun in the list below, replace it with the correct spelling, converted to the script or phonetic transcription used in the output language.\n\n- For Korean proper nouns:\n    - Always correct proper nouns to the standard spelling, then transcribe them using the script or phonetic convention typically used in the output language for foreign names, unless there is an official or widely accepted translation.\n    - Never leave proper nouns in Hangul in non-Korean texts.\n    - Examples:\n        - Use Latin letters (romanization) in English, Spanish, French, German, Italian, Portuguese, Indonesian, Dutch, Finnish, Croatian, Czech, Slovak, Polish, Hungarian, Swedish, Malay, Turkish, Tagalog, Swahili, Uzbek.\n        - Use Katakana in Japanese (e.g. ハンサンド).\n        - Use Hanzi (Chinese characters) or pinyin in Chinese (Simplified, Traditional, Cantonese) if widely accepted.\n        - Use local phonetic script in languages such as Thai, Arabic, Russian, Greek, Hebrew, Hindi, Mongolian, Persian, Ukrainian.\n        - Use Hangul in Korean.\n- Do NOT answer any questions.\n- Do NOT explain corrections.\n- Do NOT rephrase or simplify sentences.\n- Only perform necessary corrections as defined above.\n\n## PROPER NOUN LIST (STANDARD FORMS ONLY)\n{{고유단어리스트}}"
                    st.session_state.saved_user_prompt_template = "You are a meticulous proofreader for {{주제}}.\n\n## TASK\nYour only task is to correct spelling, transcription, spacing, punctuation, or typographical errors in the given text.\n\n- The input text may contain Korean, English, Chinese, Japanese, or other languages, or a mixture of them.\n- Keep the text in its original language. Do NOT translate the entire text into another language.\n- However, for Korean proper nouns:\n    - Correct them to their official spelling from the provided proper noun list.\n    - Then transcribe them using the writing system or phonetic convention typically used in the output language for foreign names, unless there is an official or widely accepted translation.\n    - Never leave proper nouns in Hangul in non-Korean texts.\n- For all other words, correct only obvious spelling or transcription mistakes.\n- Do NOT answer questions or explain corrections.\n- Do NOT paraphrase or simplify sentences.\n\n## Origin Transcription:\n{transcription}\n\n## Corrected Transcription:"
                    st.rerun()
            
            # 현재 프롬프트 미리보기
            st.markdown("#### 📋 JSON View")
            
            # System Prompt 
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("**System**")
                st.text(st.session_state.saved_system_prompt[:50] + "..." if len(st.session_state.saved_system_prompt) > 50 else st.session_state.saved_system_prompt)
            
            with col2:
                if st.button("📋", key="show_system_json", help="JSON 형식으로 보기"):
                    show_system_prompt_json()
            
            st.markdown("---")
            
            # User Prompt Template
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("**User**")
                st.text(st.session_state.saved_user_prompt_template[:50] + "..." if len(st.session_state.saved_user_prompt_template) > 50 else st.session_state.saved_user_prompt_template)
            
            with col2:
                if st.button("📋", key="show_user_json", help="JSON 형식으로 보기"):
                    show_user_prompt_json()
        
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
        
        # 평가하기 탭
        with tab3:
            st.markdown("#### 🧪 프롬프트 평가")
            st.markdown("프롬프트를 체계적으로 테스트하고 평가할 수 있습니다.")
            
            # 모드 전환 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 평가 모드로 전환", key="switch_to_evaluate", use_container_width=True):
                    st.session_state.app_mode = "evaluate"
                    st.success("평가 모드로 전환되었습니다!")
                    st.rerun()
            
            with col2:
                if st.button("🎤 STT 모드로 돌아가기", key="switch_to_stt", use_container_width=True):
                    st.session_state.app_mode = "stt"
                    st.success("STT 모드로 전환되었습니다!")
                    st.rerun()
            
            # 현재 모드 표시
            if st.session_state.app_mode == "evaluate":
                st.info("🔄 현재 평가 모드 활성화")
            else:
                st.info("🔄 현재 STT 모드 활성화")
            
            st.markdown("---")
            
            # 평가 모드 설명
            st.markdown("**📋 평가 모드 기능:**")
            st.markdown("- 📝 프롬프트 변수 관리")
            st.markdown("- 🧪 테스트 케이스 생성 및 관리") 
            st.markdown("- 🏃‍♂️ 일괄 테스트 실행")
            st.markdown("- 📊 결과 비교 및 분석")

    # 모드에 따라 다른 메인 화면 표시
    if st.session_state.app_mode == "stt":
        show_stt_interface()
    else:
        show_evaluation_interface()

def show_stt_interface():
    """STT 모드 인터페이스"""
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
            global stop_recording
            stop_recording = True
            st.rerun()  # 버튼 상태를 즉시 업데이트

    # 녹음 중일 때 음성 인식 실행
    if st.session_state.is_recording:
        with st.spinner("🎤 음성을 인식하는 중... (1.5초 멈추면 자동 종료)"):
            user_input = recognize_speech_with_interrupt()
            
        # 녹음 완료 후 처리
        st.session_state.is_recording = False
        
        if user_input and "중단되었습니다" not in user_input and "인식할 수 없습니다" not in user_input:
            process_text_input(user_input, "음성")
            st.rerun()
        elif user_input:
            if "중단되었습니다" in user_input:
                st.info("🔴 녹음이 중단되었습니다.")
            else:
                st.warning(f"⚠️ {user_input}")
        
        st.rerun()  # 버튼 상태 업데이트

    # 텍스트 입력 부분 (음성 입력 아래에 추가)
    st.markdown("#### 텍스트로 직접 입력하기")
    
    # 텍스트 입력 필드
    text_input = st.text_area("텍스트를 입력하세요:", 
                               height=100,
                               placeholder="ex. 안녕하세요")
    
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
            
        if st.session_state.get('tm_corrected_text'):
            with st.container():
                if st.session_state.corrected_text != st.session_state.tm_corrected_text:
                    # TM이 적용된 경우
                    st.markdown("**📊 TM 교정:**")
                    st.success(st.session_state.tm_corrected_text)
                else:
                    # TM이 적용되지 않은 경우
                    st.markdown("**📊 TM 교정: TM 적용되지 않음**")
                    st.success(st.session_state.corrected_text)
                
        if st.session_state.get('translated_text'):
            with st.container():
                st.markdown("**🌐 번역:**")
                st.success(st.session_state.translated_text)

def show_evaluation_interface():
    """평가 모드 인터페이스 - 클로드 프롬프트 평가기 스타일"""
    st.subheader("🧪 프롬프트 평가 및 테스트")
    
    # 상단 컨트롤 패널
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**📝 평가할 프롬프트**")
        evaluation_prompt = st.text_area(
            "프롬프트를 입력하세요 (변수는 {{변수명}} 형식으로 사용):",
            value=st.session_state.evaluation_prompt,
            height=100,
            placeholder="예: 다음 텍스트를 {{언어}}로 번역해주세요: {{텍스트}}"
        )
        st.session_state.evaluation_prompt = evaluation_prompt
    
    with col2:
        st.markdown("**⚙️ 변수 관리**")
        # 새 변수 추가
        new_var_name = st.text_input("새 변수명:", placeholder="예: 언어", key="new_var_name")
        new_var_value = st.text_input("기본값:", placeholder="예: 영어", key="new_var_value")
        
        if st.button("➕ 변수 추가", key="add_variable"):
            if new_var_name and new_var_value:
                st.session_state.prompt_variables[new_var_name] = new_var_value
                st.success(f"변수 '{new_var_name}' 추가됨!")
                st.rerun()
            else:
                st.warning("변수명과 기본값을 모두 입력해주세요.")
    
    with col3:
        st.markdown("**🚀 액션**")
        if st.button("➕ 새 테스트 케이스", key="add_test_case_btn", use_container_width=True):
            # 기본 테스트 케이스 추가
            if st.session_state.prompt_variables:
                test_case = {
                    "name": f"테스트 {len(st.session_state.test_cases) + 1}",
                    "variables": {k: v for k, v in st.session_state.prompt_variables.items()},
                    "expected": "",
                    "result": None
                }
                st.session_state.test_cases.append(test_case)
                st.success("새 테스트 케이스 추가됨!")
                st.rerun()
            else:
                st.warning("먼저 변수를 추가해주세요.")
    
    # 테이블 형태의 테스트 케이스 표시
    if st.session_state.test_cases:
        st.markdown("### 📊 테스트 케이스 테이블")
        
        # 테이블 헤더 생성
        if st.session_state.prompt_variables:
            # 변수 컬럼들
            var_columns = list(st.session_state.prompt_variables.keys())
            
            # 테이블 데이터 준비
            table_data = []
            for i, test_case in enumerate(st.session_state.test_cases):
                row = {
                    "테스트 케이스": test_case['name'],
                    "예상 결과": test_case.get('expected', ''),
                    "실행 결과": test_case.get('result', ''),
                    "상태": "✅ 완료" if test_case.get('result') else "⏳ 대기"
                }
                
                # 변수 값들 추가
                for var_name in var_columns:
                    row[var_name] = test_case['variables'].get(var_name, '')
                
                table_data.append(row)
            
            # DataFrame으로 변환하여 표시
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # 테이블 아래 액션 버튼들
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏃‍♂️ 모든 테스트 실행", key="run_all_tests", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, test_case in enumerate(st.session_state.test_cases):
                        status_text.text(f"테스트 '{test_case['name']}' 실행 중...")
                        
                        # 프롬프트에 변수 적용
                        filled_prompt = st.session_state.evaluation_prompt
                        for var_name, var_value in test_case['variables'].items():
                            filled_prompt = filled_prompt.replace(f"{{{{{var_name}}}}}", var_value)
                        
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "user", "content": filled_prompt}
                                ],
                                max_tokens=500,
                                temperature=0.3
                            )
                            result = response.choices[0].message.content.strip()
                            st.session_state.test_cases[i]['result'] = result
                        except Exception as e:
                            st.session_state.test_cases[i]['result'] = f"오류: {e}"
                        
                        progress_bar.progress((i + 1) / len(st.session_state.test_cases))
                    
                    status_text.text("모든 테스트 완료!")
                    st.success("🎉 모든 테스트가 완료되었습니다!")
                    st.rerun()
            
            with col2:
                if st.button("📊 결과 내보내기", key="export_results", use_container_width=True):
                    # CSV로 내보내기
                    export_df = pd.DataFrame(table_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv,
                        file_name="prompt_evaluation_results.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🗑️ 모든 테스트 삭제", key="clear_all_tests", use_container_width=True):
                    st.session_state.test_cases = []
                    st.success("모든 테스트 케이스가 삭제되었습니다!")
                    st.rerun()
            
            # 개별 테스트 케이스 관리
            st.markdown("### 📝 개별 테스트 케이스 관리")
            for i, test_case in enumerate(st.session_state.test_cases):
                with st.expander(f"📋 {test_case['name']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # 테스트 케이스 이름 수정
                        new_name = st.text_input(
                            "테스트 케이스 이름:",
                            value=test_case['name'],
                            key=f"test_name_{i}"
                        )
                        if new_name != test_case['name']:
                            st.session_state.test_cases[i]['name'] = new_name
                        
                        # 변수 값들 수정
                        st.markdown("**변수 값들:**")
                        for var_name in st.session_state.prompt_variables.keys():
                            new_value = st.text_input(
                                f"{var_name}:",
                                value=test_case['variables'].get(var_name, ''),
                                key=f"test_var_{i}_{var_name}"
                            )
                            if new_value != test_case['variables'].get(var_name, ''):
                                st.session_state.test_cases[i]['variables'][var_name] = new_value
                        
                        # 예상 결과 수정
                        expected = st.text_area(
                            "예상 결과:",
                            value=test_case.get('expected', ''),
                            key=f"test_expected_{i}",
                            height=60
                        )
                        if expected != test_case.get('expected', ''):
                            st.session_state.test_cases[i]['expected'] = expected
                    
                    with col2:
                        # 개별 실행 버튼
                        if st.button("🏃‍♂️ 실행", key=f"run_test_{i}"):
                            # 프롬프트에 변수 적용
                            filled_prompt = st.session_state.evaluation_prompt
                            for var_name, var_value in test_case['variables'].items():
                                filled_prompt = filled_prompt.replace(f"{{{{{var_name}}}}}", var_value)
                            
                            # AI 호출
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "user", "content": filled_prompt}
                                    ],
                                    max_tokens=500,
                                    temperature=0.3
                                )
                                result = response.choices[0].message.content.strip()
                                st.session_state.test_cases[i]['result'] = result
                                st.rerun()
                            except Exception as e:
                                st.error(f"테스트 실행 실패: {e}")
                        
                        # 삭제 버튼
                        if st.button("🗑️ 삭제", key=f"del_test_{i}"):
                            st.session_state.test_cases.pop(i)
                            st.rerun()
                    
                    # 실행 결과 표시
                    if test_case.get('result'):
                        st.markdown("**실행 결과:**")
                        st.success(test_case['result'])
    else:
        st.info("아직 테스트 케이스가 없습니다. 변수를 추가하고 '새 테스트 케이스' 버튼을 클릭해보세요!")


if __name__ == "__main__":
    main()
