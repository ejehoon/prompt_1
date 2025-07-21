import streamlit as st
import pandas as pd
import openai
import time
import os

try:
    from st_audiorec import st_audiorec
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    try:
        from streamlit_audio_recorder import audio_recorder
        AUDIO_RECORDER_AVAILABLE = True
    except ImportError:
        AUDIO_RECORDER_AVAILABLE = False

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

def transcribe_audio_with_whisper(audio_bytes):
    """OpenAI Whisper를 사용하여 오디오를 텍스트로 변환"""
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
        tm_status = "✅ TM 교정 적용됨" if st.session_state.recognized_text != st.session_state.tm_corrected_text else "➖ TM 교정 변경사항 없음"
        debug_info["TM 정보"] = f"📊 TM 항목 수: {len(st.session_state.tm_df)}개\n{tm_status}"
    
    st.session_state.debug_info = debug_info


def main():
    st.set_page_config(
        page_title="STT 교정 테스트",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 STT 교정 테스트")
    st.markdown("**iPad 및 웹 환경 호환 버전**")

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

    # 메인 영역 - 입력 및 처리
    col1, col2 = st.columns([1, 1])
    
    # 세션 상태 초기화
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = None
    if 'tm_corrected_text' not in st.session_state:
        st.session_state.tm_corrected_text = None
    if 'corrected_text' not in st.session_state:
        st.session_state.corrected_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None

    with col1:
        st.subheader("🎤 음성 입력")
        st.markdown("**iPad 및 모바일 지원**")
        
        if not AUDIO_RECORDER_AVAILABLE:
            st.error("⚠️ 오디오 녹음 패키지가 설치되지 않았습니다.")
            st.info("수동으로 오디오 파일을 업로드할 수 있습니다:")
            
            # 파일 업로드로 대체
            uploaded_audio = st.file_uploader(
                "오디오 파일 업로드", 
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="녹음된 오디오 파일을 업로드하세요."
            )
            
            if uploaded_audio is not None:
                st.audio(uploaded_audio)
                
                if st.button("🔍 음성 인식", key="transcribe_uploaded", use_container_width=True):
                    with st.spinner("🎤 OpenAI Whisper로 음성을 인식하는 중..."):
                        # 업로드된 오디오 파일 처리
                        audio_bytes = uploaded_audio.read()
                        
                        # Whisper API로 전사
                        transcribed_text = transcribe_audio_with_whisper(audio_bytes)
                        
                        if transcribed_text:
                            process_text_input(transcribed_text, "음성(Whisper)")
                            st.success(f"✅ 음성 인식 완료: {transcribed_text}")
                            st.rerun()
                        else:
                            st.error("❌ 음성 인식에 실패했습니다.")
        else:
            # 첫 번째 패키지 시도
            try:
                audio_data = st_audiorec()
                
                if audio_data is not None:
                    st.audio(audio_data, format='audio/wav')
                    
                    if st.button("🔍 음성 인식", key="transcribe_button1", use_container_width=True):
                        with st.spinner("🎤 OpenAI Whisper로 음성을 인식하는 중..."):
                            # Whisper API로 전사
                            transcribed_text = transcribe_audio_with_whisper(audio_data)
                            
                            if transcribed_text:
                                process_text_input(transcribed_text, "음성(Whisper)")
                                st.success(f"✅ 음성 인식 완료: {transcribed_text}")
                                st.rerun()
                            else:
                                st.error("❌ 음성 인식에 실패했습니다.")
            except NameError:
                # 두 번째 패키지 시도
                try:
                    audio_bytes = audio_recorder()
                    
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        
                        if st.button("🔍 음성 인식", key="transcribe_button2", use_container_width=True):
                            with st.spinner("🎤 OpenAI Whisper로 음성을 인식하는 중..."):
                                # Whisper API로 전사
                                transcribed_text = transcribe_audio_with_whisper(audio_bytes)
                                
                                if transcribed_text:
                                    process_text_input(transcribed_text, "음성(Whisper)")
                                    st.success(f"✅ 음성 인식 완료: {transcribed_text}")
                                    st.rerun()
                                else:
                                    st.error("❌ 음성 인식에 실패했습니다.")
                except NameError:
                    st.error("⚠️ 오디오 녹음 라이브러리를 사용할 수 없습니다.")
                    st.info("텍스트 입력을 사용해주세요.")
    
    with col2:
        st.subheader("✏️ 텍스트 입력")
        st.markdown("**직접 텍스트 입력**")
        
        # 텍스트 입력 필드
        text_input = st.text_area(
            "텍스트를 입력하세요:", 
            height=150,
            placeholder="예: 안녕하세요. 처리하기를 눌러주세요.",
            key="text_input_main"
        )
        
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
                
        # 결과 지우기 버튼
        st.markdown("---")
        if st.button("🗑️ 전체 지우기", key="clear_all", use_container_width=True):
            st.session_state.recognized_text = None
            st.session_state.tm_corrected_text = None
            st.session_state.corrected_text = None
            st.session_state.translated_text = None
            st.success("모든 결과가 삭제되었습니다!")
            st.rerun()


if __name__ == "__main__":
    main()
