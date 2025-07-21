# 🎤 STT 교정 테스트 앱

음성을 텍스트로 변환하고, TM(Translation Memory)과 LLM을 활용하여 교정 및 번역하는 Streamlit 애플리케이션입니다.

## 📋 기능

- 🎤 **음성 인식**: Google Speech Recognition을 이용한 한국어 음성 텍스트 변환
- 📝 **텍스트 입력**: 직접 텍스트 입력 및 처리
- 📊 **TM 교정**: 번역 메모리를 활용한 용어 교정
- 🔍 **LLM 검수**: OpenAI GPT를 이용한 텍스트 검수
- 🌐 **자동 번역**: 검수된 텍스트의 영어 번역
- 🛠️ **프롬프트 커스터마이징**: System/User 프롬프트 편집 가능

## 🚀 배포하기

### 1. GitHub에 코드 업로드

```bash
git clone <your-repo-url>
cd <your-repo-name>
# 파일들을 업로드하고 커밋
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Streamlit Cloud에서 배포

1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. Repository, Branch, Main file path 설정
   - Main file path: `prompt.py`
5. "Deploy!" 클릭

### 3. API 키 설정

배포 후 **App settings** → **Secrets**에서 다음과 같이 설정:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

## 📦 필요한 파일들

- `prompt.py` - 메인 애플리케이션
- `requirements.txt` - Python 패키지 의존성
- `packages.txt` - 시스템 패키지 (오디오 기능용)

## 🔧 로컬 실행

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
streamlit run prompt.py
```

## ⚠️ 주의사항

- API 키는 절대 GitHub에 업로드하지 마세요
- Streamlit Cloud의 Secrets 기능을 사용하여 안전하게 관리하세요
- 마이크 기능은 HTTPS 환경에서만 작동합니다 