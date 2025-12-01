# LLM 기반 인터랙티브 교육·제어 시스템 프로젝트

## 1. 프로젝트 개요 *필수작성*

* **수행 학기:** 2학기
* **프로젝트명:** LLM 기반 인터랙티브 교육·제어 시스템 개발
* **작성날짜:** 2025.11.17

| 구분 | 성명  | 학번       | 소속학과      | 깃허브 아이디      |
| -- | --- | -------- | --------- | ------------ |
| 1  | 김건우 | 20201702 | 컴퓨터학과     | rjsdn2308    |
| 2  | 강민수 | 20231901 | 데이터사이언스학과 | MoriartyKang |
| 3  | 이준성 | 20211704 | 데이터사이언스학과 | junsung2001  |

* **지도교수:** 화성의과학대학교 김정은 교수

---

## 2. 프로젝트 내용

### 2.1 서론

최근 초등학교 저학년 아이들을 둔 부모들의 온라인 커뮤니티를 살펴보면,
“아이들이 과일이 어디에서 자라는지 잘 모른다”, “마트에서 바로 나오는 줄 안다”는 글에 공감을 많이 하는 글을 보고 아이디어가 떠올랐습니다.

과일의 모양, 성장 과정, 재배 환경을 자연스럽게 배울 기회가 적어지면서 과일·식물 학습이 실생활과 분리되는 현상이 나타났습니다.

이에 본 프로젝트는 아이들이 과일의 종류·특징·자라는 방식을 쉽고 재미있게 배울 수 있는 유아 친화형 교육 도구를 개발하는 것을 목표로 합니다.

프로젝트 핵심은 두 가지입니다.

* 이미지를 통한 과일 인식(CV)
* 아이 눈높이 설명 + 퀴즈 제공 LLM 기반 챗봇

이를 결합한 앱 형태로, 아이들이 직접 과일 사진을 찍고,
“이 과일은 무엇이고 어디에서 자라는지” 자연스럽게 학습할 수 있도록 설계했습니다.

---

### 2.2 추진 배경 (자료 조사 및 요구 분석)

#### 2.2.1 개발 배경 및 필요성

아이들이 과일을 접할 기회는 많지만, 실제 성장 과정이나 자라는 방식을 보는 기회는 부족

부모 커뮤니티와 교육 블로그에서 나타난 필요성:

* “아이에게 수박이 땅에서 자란다고 설명해도 실제 모습을 보여주기 어렵다.”
* “사과가 나무에서 열린다는 개념을 잘 모른다.”
* “마트 상품으로만 과일을 인식하는 경향이 있다.”

이에 따라 경험형 학습 도구 필요성이 증가했습니다.

본 프로젝트는 다음 요소를 하나의 앱으로 통합했습니다:

* 과일 사진을 분석해주는 CV 기술
* 아이 눈높이 설명을 제공하는 LLM
* O/X 퀴즈를 통한 학습 강화

#### 2.2.2 선행 기술 및 사례 분석
(1) 과일 인식(CV) 기반 서비스
Fruits-360 등과 같은 공개 데이터셋 기반 연구는 존재하나 대부분 연구용 데이터로, 실제 유아 학습 도구는 매우 제한적
식별 정도는 가능하지만 “수확 전 모습”, “자라는 방식(tree/ground/vine)”까지 안내하는 서비스는 드문 편

(2) 교육용 LLM 활용 사례
GPT API를 이용한 교육용 Q&A 서비스는 증가하는 추세
그러나 유아 맞춤 말투 + 간단한 설명 + OX 퀴즈까지 통합한 사례는 거의 없음
대부분의 LLM 챗봇은 어린이에게 너무 어렵거나 장문 위주의 설명 제공

(3) 본 프로젝트의 차별성
1.CV와 LLM을 결합한 유아 맞춤형 앱
2.아이에게 쉬운 말투·짧은 문장·친절한 안내 사용
3.수확 전 과일 이미지 제공 (시각 기반 학습 강화)
4.OX 퀴즈까지 포함한 반복 학습 구조
5.파스텔 톤 색상 및 단순 UI로 유아 친화적 환경 제공

### 2.2.3 선행 기술 한계 및 개선점
기존 한계 -> 본 프로젝트 개선 내용
과일 데이터셋이 연구용에 국한됨 -> 직접 수집한 실제 사진 기반 과일 인식 모델 구축
어린이에게 설명하기 어려운 LLM 말투 -> 유아 수준에 맞춘 부드러운 설명 프롬프트 설계
단순한 이미지 분류만 제공 -> 자라는 방식(tree/ground/vine) + 수확 전 이미지 + 설명 제공
학습 요소 부족 -> O/X 퀴즈, 아이 눈높이 감성 재작성 기능 포함

---

### 2.3 목표 및 내용

#### 2.3.1 프로젝트 목표

1. 사진 기반 과일 종류 자동 인식
2. 과일 자라는 방식 및 모습 학습 지원
3. LLM 활용한 아이 눈높이 설명 제공
4. 간단한 과일 O/X 퀴즈 제공
5. 파스텔 기반 UI를 갖춘 유아 친화적 학습 앱 제작

#### 2.3.2 개발 범위

| 구분            | 내용                       |
| ------------- | ------------------------ |
| 과일 이미지 분석(CV) | ResNet18 기반 과일 분류 모델 구현  |
| LLM 기반 설명 생성  | OpenAI API 기반 유아용 설명 생성  |
| 퀴즈 생성 엔진      | O/X 과일 퀴즈 생성 및 정답 판단     |
| UI/UX         | Gradio 기반 유아 친화적 인터페이스   |
| 데이터 구조        | fruit_meta.json 기반 정보 관리 |

#### 2.3.3 시스템 구조 블록 다이어그램

사용자 → 과일 사진 업로드 → CV 모델 → 과일 종류/정보 → LLM 설명 생성 → 유아용 말투로 설명 → 사용자 학습

#### 2.3.4 시퀀스 다이어그램

User → Web UI → 사진 업로드
↓
CV 모델 분석 → 과일 종류 예측
↓
메타 정보 조회 → grow_type / 수확 이미지
↓
LLM → 아이에게 맞춘 설명 생성
↓
UI에 결과 출력

#### 2.3.5 개발 환경 및 구현 결과

* 개발 환경: Python 3.10, PyTorch(ResNet18), Gradio, OpenAI API, JSON 기반 메타데이터
* 주요 기능 구현 결과:

  * 과일 인식 모델 학습(apple, banana, strawberry)
  * 수확 전 과일 이미지 연결 기능 구현
  * 아이 맞춤형 설명 생성 LLM 프롬프트 구축
  * 유아 친화적 UI (파스텔 기반 CSS)
  * 과일 O/X 퀴즈 기능 정상 작동
  * 챗봇 모드에서 자연스러운 아이 눈높이 대화 가능
  * CV+LLM 결합 흐름 정상 동작

#### 2.3.6 문제점 및 한계점

* 과일 이름이 같아도 국가/종류에 따라 모양이 다름 → 모델 일반화 필요
* 과일 사진 데이터 부족 → 저작권 없는 데이터 수집·정제 시간 필요
* 다양한 촬영 환경 반영 데이터 부족 → 실외·조명·부분 가림 문제
* LLM 호출 비용 문제 → 무료 서비스용 캐싱·경량 모델 필요

---

### 📌 (3) 손 제스처 기반 스마트 제어

#### 개요

손 제스처를 인식하여 **스마트홈 기기 제어**와 **LLM 기반 자연스러운 안내 메시지 생성**을 수행합니다. MediaPipe HandLandmarker 모델을 사용하여 손가락 위치를 분석하고, 제스처(open palm, fist, thumb up)에 따라 조명, 사운드 등 IoT 상태를 제어합니다.

#### 기술 스택

* HTML5, CSS, JavaScript
* MediaPipe HandLandmarker (Vision Tasks)
* WebRTC 기반 카메라 입력
* Web Audio API (비프음 재생)
* Canvas API (손 랜드마크 시각화)

#### 주요 코드 및 분석

```javascript
// MediaPipe HandLandmarker 초기화
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
);
const handLandmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions:{
    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  },
  runningMode: "VIDEO",
  numHands: 1
});
```

```javascript
// 손 랜드마크 21점 분석 후 제스처 판별 함수
const TIP = { TH:4, IX:8, MID:12, RIN:16, PIN:20 };
const PIP = { IX:6, MID:10, RIN:14, PIN:18 };

const countExtended = lm =>
  [ [TIP.IX,PIP.IX],[TIP.MID,PIP.MID],[TIP.RIN,PIP.RIN],[TIP.PIN,PIP.PIN] ]
  .reduce((c,[t,p])=> c + (lm[t].y < lm[p].y), 0);

const isFist = lm =>
  [ [TIP.IX,PIP.IX],[TIP.MID,PIP.MID],[TIP.RIN,PIP.RIN],[TIP.PIN,PIP.PIN] ]
  .reduce((c,[t,p])=> c + (lm[t].y > lm[p].y), 0) >= 3;

const isOpen = lm => countExtended(lm) >= 3;

const isThumbUp = lm => {
  const dx = lm[4].x - lm[2].x;
  const dy = lm[4].y - lm[2].y;
  return (dy < -0.03 && Math.abs(dx) < 0.10);
};

const classify = lm => isFist(lm) ? 'FIST' :
                        isThumbUp(lm) ? 'THUMB_UP' :
                        isOpen(lm) ? 'OPEN' : 'UNKNOWN';
```

```javascript
// Web Audio API를 이용한 비프음 제어
let osc=null, gain=null, playing=false;

function audioPlay(){
  if(playing) return;
  osc = audioCtx.createOscillator();
  gain = audioCtx.createGain();
  osc.type = 'sine';
  osc.frequency.value = 440;
  gain.gain.value = 0.05;
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  playing = true;
  audioEl.textContent = 'PLAYING';
}

function audioPause(){
  if(!playing) return;
  try{ osc.stop(); }catch(_){}
  osc.disconnect(); gain.disconnect();
  playing=false;
  audioEl.textContent='PAUSED';
}

function audioToggle(){ playing ? audioPause() : audioPlay(); }
```

```javascript
// 제스처 판별에 따른 행동 매핑
if(cnt>=CONFIRM){
  if(g==='OPEN') setLight(true);      // 손바닥 펼치기 → 조명 켜기
  else if(g==='FIST') setLight(false); // 주먹 → 조명 끄기
  else if(g==='THUMB_UP') audioToggle(); // 엄지 up → 사운드 토글
  cnt=0;
}
```

#### 장점 및 단점

* 장점

  * 직관적 제스처 기반 제어 가능
  * 실시간 피드백 + 시각화 제공
  * LLM 안내와 결합 가능 (자연스러운 안내문 생성)
* 단점

  * 조명/음향 장치 제어는 브라우저 환경 제한
  * 다양한 손 위치/조명 환경에서 인식 정확도 편차 발생
  * LLM 연동 시 API 호출 비용 발생

---

### 2.4 기대효과

#### 교육적 효과

* 아이 스스로 사진 찍으며 탐구하는 경험형 학습 가능
* 과일 자라는 방식·모습 자연스러운 이해 지원

#### 기술적/사회적 효과

* 부모·교사 교육 효율 향상
* CV+LLM 결합 학습 도구로 기술적 확장성 확보
* 손 제스처 기반 제어 경험 제공 → IoT 분야 확장 가능

---

### 2.6 역할 분담

|  구분 |  성명 | 팀내 역할                                                                                                                     |
| :-: | :-: | ------------------------------------------------------------------------------------------------------------------------- |
|  1  | 강민수 | CV 모델 개발, 앱 연동, 과일 이미지 수집/정제, ResNet18 모델 학습/검증, Gradio UI 연동, 수확 전 이미지 매핑/fruit_meta.json 설계                             |
|  2  | 김건우 | LLM 기능 구현, 유아 친화형 대화 설계, LLM 프롬프트 설계, OX 퀴즈 생성/정답 분석, 과일 Q&A 챗봇 흐름 구축, CV 결과 기반 상세 설명 연결, 손 제스처 제어 HTML/JS 분석 및 LLM 안내 연동 |
|  3  | 이준성 | 전체 앱 구조·UI/UX 통합, Gradio 기반 UI 구성·파스텔 톤 디자인, CV–LLM–퀴즈 기능 통합, 상태 관리, 이미지 분석/설명/대화 흐름 연결, 배포용 구조 정리                        |

---

### 2.7 참고문헌

1. Mureșan & Oltean, Fruit recognition from images using deep learning, 2017
2. Fruits-360 Dataset, Kaggle, 2024
3. Yang et al., Lightweight and Efficient Deep Learning Models for Fruit Classification, 2024
4. OpenAI API Documentation, 2025
5. Gradio Documentation, 2025
6. MediaPipe HandLandmarker Documentation, 2025
