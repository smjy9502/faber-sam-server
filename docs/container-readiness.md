# P2 — Container Readiness

## 현재 검증 범위

P2는 소스와 실행 recipe 준비 단계다. 컨테이너 build/run, 의존성 설치,
모델 다운로드, 실제 추론, 클라우드 배포는 실행하지 않았다.
아래 명령은 P3 승인 후 사용할 **미실행 예시**다.

기존 성공 API 계약은 multipart 입력과 PNG base64 JSON 응답으로 유지한다.
`/segment`의 point / box / positive·negative 배열과 mask/score/candidates/prompt,
`/remove-background`의 mask/cutout/engine/model_id/bbox 필드를 유지한다.
alpha는 여전히 `min(original, model)`이고 원본 RGB를 보존한다.
SAM candidate ranking과 bbox/metric 계산은 기존 코드를 유지한다.

## 실행 환경 / 의존성

- 컨테이너 후보: Python 3.11.14, Debian bookworm slim, Linux/amd64.
  3.9의 노후화와 매우 최신 Python wheel 호환 위험을 피한 선택이다.
- `requirements.txt`: 기존 Mac용 파일 그대로 유지.
- `requirements-container.txt`: CPU 전용 direct dependency exact pins.
  torch 2.8.0+cpu / torchvision 0.23.0+cpu는 공식 호환 조합이다.
- SAM commit은 기존 로컬 설치 metadata의
  `dca509fe793f601edb92606367a655c15ac00fdf`와 동일하다.
- 기존 `.venv` metadata: torch 2.8.0, torchvision 0.23.0,
  FastAPI 0.128.8, Starlette 0.49.3, Uvicorn 0.39.0, Pillow 11.3.0.
  컨테이너 NumPy 2.2.6/OpenCV headless 4.12 및 BiRefNet dependency 조합은 후보이며
  Linux에서 아직 설치·실행 검증하지 않았다.
- direct pin은 transitive hash lock이 아니다. P3에서 resolver/pip check와
  Linux smoke를 통과한 뒤 전체 의존성 lock과 base image digest를 기록한다.
- builder에만 git/CA 인증서 설치, runtime에는 builder의 venv만 복사한다.
  빌드 자체는 네트워크·디스크 사용을 수반하며 P2에서 실행하지 않는다.

## 모델 준비

SAM은 import 시점 대신 ASGI lifespan startup에서 로딩한다. checkpoint가 없으면
`SAM_CHECKPOINT_MISSING`으로 시작에 실패한다. 모델 다운로드 fallback은 없다.

BiRefNet은 첫 배경제거 요청에서 lazy load한다.
`BIREFNET_MODEL_PATH`가 비어 있으면 기존 Hugging Face cache에서
`ZhengPeng7/BiRefNet`의 고정 revision을 찾는다.
`revision`과 `code_revision` 모두
`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`로 고정한다.
`local_files_only=True`이며 컨테이너는 HF/Transformers offline env도 사용한다.
cache가 없으면 `MODEL_UNAVAILABLE`이며 자동 다운로드하지 않는다.
로컬 환경도 같은 정책이므로 기존 cache가 없는 새 개발 환경에는 사전 모델 준비가 필요하다.

local packaged 폴더는 weight뿐 아니라 config 및 trusted Python code를 포함해야 한다.
로컬 폴더의 파일은 revision 옵션만으로 무결성이 검증되지 않는다.
P3에서 신뢰한 고정 snapshot과 checksum manifest를 검증해야 한다.
모델 코드 내부의 추가 다운로드 시도 여부도 offline 실제 로딩 smoke로 검증한다.
모델 bake/GCS/외부 cache의 최종 선택은 보류한다.

`.dockerignore`는 source allowlist이며 `models/`, `.venv`, `.git`, `.env*`,
fixture/PNG/PDF/테스트 출력은 전부 제외한다. Dockerfile은 weight를 복사하지 않는다.
따라서 현재 recipe만 빌드한 이미지는 모델 공급 없이 기동하지 않는다.
가짜 download step이나 dummy 모델은 포함하지 않는다.

## P3용 명령 예시 — 아직 미실행

기존의 신뢰한 모델을 사용할 준비와 container runtime 설치가 선행되어야 한다.
다음 `/ABSOLUTE/PREPARED_MODELS`는 SAM checkpoint와 고정 BiRefNet 전체 snapshot을
담은 디렉터리의 예시이며 P2에서 복사·생성하지 않았다.

```sh
docker build --platform linux/amd64 -t faber-inference:p3 .
docker run --rm --platform linux/amd64 -p 127.0.0.1:8000:8080 \
  --mount type=bind,source=/ABSOLUTE/PREPARED_MODELS,target=/models,readonly \
  -e CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000 \
  faber-inference:p3
```

ARM Mac의 amd64 emulation 수치는 실제 Cloud Run CPU 성능과 구분한다.
기존 native 환경은 기존 uvicorn 명령으로도 동작하나, production-safe logging과
worker/PORT 계약은 `PORT=8000 python run_server.py` 실행 경로를 사용한다.

## 설정 — .env 생성 없음

| 환경변수 | Native 기본값 | Docker 기본값 / 정책 |
|---|---|---|
| PORT | 8080 | 8080, `0.0.0.0`, worker 1 |
| SAM_CHECKPOINT_PATH | repo/models/sam_vit_b_01ec64.pth | /models/sam_vit_b_01ec64.pth |
| BIREFNET_MODEL_PATH | 빈 값, 고정 HF cache | /models/birefnet |
| BIREFNET_REVISION | 위 고정 SHA | 동일, 변경하려면 full SHA 필수 |
| MAX_UPLOAD_BYTES | 20971520 (20 MiB) | 같은 상한 |
| MAX_REQUEST_BYTES | 22020096 (21 MiB) | multipart overhead 1 MiB |
| MAX_WIDTH / MAX_HEIGHT | 각 8192 | 같은 상한 |
| MAX_PIXELS | 24000000 | 같은 상한 |
| MAX_POINTS | positive/negative 각각 64 | 총 최대 128 |
| MAX_PROMPT_BYTES | 각 16384 | multipart text part 및 JSON 상한 |
| MAX_RESPONSE_BYTES | 29360128 (28 MiB) | 32 MiB보다 4 MiB 여유 |
| TORCH_THREADS | 2 | 2, interop 1 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | torch 설정 사용 | 각각 2 |
| LOG_LEVEL | INFO | INFO/WARNING/ERROR만 허용 |
| CORS_ORIGINS | localhost:3000,127.0.0.1:3000의 http origin | 빈 allowlist |

hard limit은 더 낮출 수 있지만 코드의 안전 상한보다 높일 수 없다.
PE의 20 MiB/24 MP는 UX heuristic이고 여기서는 별개의 서버 강제 제한이다.
20 MiB 파일과 21 MiB 전체 request는 Cloud Run 32 MiB HTTP/1 request보다 작다.
24 MP RGBA buffer 하나가 약 96 MB이며 복사본·모델 activation은 추가로 필요하다.
최대 한 변 제한은 극단적 입력을 제한한다. RAM 안전성은 P3에서 측정한다.
PNG/JPEG/WebP의 실제 signature와 decode format이 일치해야 한다. 애니메이션은 거부한다.
좌표는 유한수이며 0..width-1 / 0..height-1, box 1개·4좌표가 필수다.
정상 범위를 벗어난 값을 조용히 clamp하지 않는다. 중복 필드도 거부한다.

PNG 인코더는 binary byte budget을 적용하며 base64 길이를 **변환 전에** 계산한다.
모든 이미지의 누적 크기를 검사하고 JSON용 64 KiB를 예약한다.
최종 JSON도 실제 UTF-8 byte 크기를 검사한 동일 bytes로 전송한다.
향후 binary/Storage 응답 전환은 별도 계약 변경이다.

## 동시성 / lifecycle

SAM set_image/predict 전체가 동일 process lock 안에 있다.
BiRefNet 추론도 이 lock을 공유하여 두 모델의 동시 peak를 줄인다.
별도 lazy resource lock이 최초 BiRefNet 로딩을 1회로 제한한다.
실패 후 사용자의 다음 요청은 재시도할 수 있지만 내부 자동 retry는 없다.
프로세스 간 공유 memory는 없으며 각 worker는 자기 모델과 lock을 갖는다.
entrypoint는 worker 1을 고정한다. 후속 Cloud Run도 concurrency 1을 권장한다.
lock은 비용 quota나 대기열 상한이 아니다. 다수 동시 요청은 buffer/RAM을 점유할 수 있다.

`/health`: 프로세스 생존, SAM ready와 BiRefNet loaded를 표시한다. 모델을 로딩하지 않는다.
`/ready`: SAM ready이면 200, 아니면 503. BiRefNet lazy 미로딩은 readiness 실패가 아니다.
이는 BiRefNet의 첫 요청 성공을 보장하지 않는다. P3에서 별도 warm smoke가 필요하다.
startup probe에는 `/ready`를 사용하고 timeout/threshold는 실측 후 정한다.

HTTP 연결이 끊기거나 timeout이어도 Python thread/torch 연산이 자동 취소되지 않는다.
무리한 thread kill은 구현하지 않았다. P3에서 cold/warm p95와 queue 시간을 측정해
client/Next/Cloud Run timeout을 맞추고 중복 retry를 막는다.

## 오류 / 로그 / 인증

성공 응답 필드는 유지한다. 기존 처리 실패는 HTTP 200 + `ok:false`를 유지하면서
안전한 `error`와 stable `code`를 반환한다. 신규 size 제한은 413,
malformed multipart는 422, 모델 준비 실패는 503, 경계 처리 실패는 500이다.
내부 예외 문자열·경로·stack trace는 응답이나 앱 로그에 넣지 않는다.
production entrypoint는 Uvicorn 오류도 일반 이벤트로 정제하고 access log를 끈다.
원본 이미지·mask/base64·좌표·filename·header/token은 로그에서 제외한다.
request ID는 서버가 새로 만들고 endpoint/status/latency/dimensions/bytes/model 상태만 기록한다.
정상 health probe는 앱 로그를 남기지 않는다.

Local CORS는 기존 두 origin을 유지한다. Docker는 기본 CORS 없음이며 wildcard는 거부한다.
API key 인증은 추가하지 않았다. Cloud Run IAM과 Next ID token proxy는 후속 단계다.
이 이미지를 unauthenticated public service로 배포하면 안 된다.

## 비용 및 후속 승인

P2는 cloud resource 생성/변경, build/push/deploy, 모델 다운로드/추론을 하지 않았다.
조기 payload reject, worker 1, 직렬화, offline 모델, 자동 retry 없음,
간결한 로그는 향후 낭비를 줄인다. 잘못된 큰 응답은 추론 후에야 알 수 있어
이미 소비한 compute 비용까지 없애지는 못한다.
Cloud Build, Registry 저장·스캔, model storage, logs, egress, idle instance,
실패·retry에는 이후 비용이 생길 수 있다. budget alert는 hard cap이 아니다.
일일 cap/rate limit/kill switch와 DEV 예산은 배포 전 별도 승인한다.

## 테스트

기존 dependency가 준비된 Python에서 다음만 실행한다.

```sh
python -B -m unittest -v test_server_hardening test_alpha_preservation
```

`test_segment.py`는 실제 localhost 모델 호출 및 출력 파일 쓰기가 있으므로 P2에서 실행 금지.
테스트는 synthetic image + fake runtime/loader/predictor, 실제 FastAPI ASGI routing을 사용한다.
이번 환경은 bundled Python 3.12와 기존 `.venv`의 pure-Python FastAPI/Starlette/multipart를
조합해 설치 없이 테스트했다. 이것은 Linux dependency resolver/이미지 검증의 대체가 아니다.

P3 전: container runtime 선택·승인, 모델 mount 준비, Linux dependency resolve,
base image digest/전체 lock, cold/warm CPU latency·RAM·response size 측정 계획 필요.
CPU/GPU 최종 선택·Cloud Run 크기·production 예산/리전·모델 packaging은 아직 미결정이다.
