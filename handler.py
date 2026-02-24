import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Base64 에러 처리를 위해 import
import subprocess
import time
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
COMFY_HTTP_READY = False

# Speed-oriented defaults/caps (override via environment if needed).
DEFAULT_LENGTH = int(os.getenv("DEFAULT_LENGTH", "65"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8"))
DEFAULT_CONTEXT_FRAMES = int(os.getenv("DEFAULT_CONTEXT_FRAMES", "49"))
DEFAULT_CONTEXT_OVERLAP = int(os.getenv("DEFAULT_CONTEXT_OVERLAP", "16"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "65"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_CONTEXT_OVERLAP = int(os.getenv("MAX_CONTEXT_OVERLAP", "24"))
ENFORCE_SPEED_LIMITS = os.getenv("ENFORCE_SPEED_LIMITS", "1").strip().lower() in ("1", "true", "yes", "on")


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value, low, high):
    return max(low, min(value, high))


def to_nearest_multiple_of_16(value):
    """주어진 값을 가장 가까운 16의 배수로 보정, 최소 16 보장"""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height 값이 숫자가 아닙니다: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted
def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수"""
    if input_type == "path":
        # 경로인 경우 그대로 반환
        logger.info(f"📁 경로 입력 처리: {input_data}")
        return input_data
    elif input_type == "url":
        # URL인 경우 다운로드
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        # Base64인 경우 디코딩하여 저장
        logger.info(f"🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")

        
def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        # wget을 사용하여 파일 다운로드
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        # Base64 문자열 디코딩
        decoded_data = base64.b64decode(base64_data)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 파일로 저장
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def _extract_first_video_base64(history, preferred_node_ids=("131",)):
    outputs = history.get("outputs", {}) if isinstance(history, dict) else {}

    ordered_node_ids = []
    for node_id in preferred_node_ids:
        if node_id in outputs:
            ordered_node_ids.append(node_id)
    for node_id in outputs.keys():
        if node_id not in ordered_node_ids:
            ordered_node_ids.append(node_id)

    for node_id in ordered_node_ids:
        node_output = outputs.get(node_id) or {}
        for key in ("gifs", "videos", "files"):
            for item in node_output.get(key, []) or []:
                fullpath = item.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    with open(fullpath, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")

                filename = item.get("filename")
                if filename:
                    subfolder = item.get("subfolder", "")
                    folder_type = item.get("type", "output")
                    try:
                        video_bytes = get_image(filename, subfolder, folder_type)
                        if video_bytes:
                            return base64.b64encode(video_bytes).decode("utf-8")
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch output via /view "
                            f"(filename={filename}, subfolder={subfolder}, type={folder_type}): {e}"
                        )
    return None


def get_video_base64(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']

    # Prefer WS completion signal, but tolerate WS drops by falling back to history polling.
    ws_failed = False
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
    except Exception as e:
        ws_failed = True
        logger.warning(f"WebSocket receive failed; falling back to history polling: {e}")

    poll_timeout_s = int(os.getenv("HISTORY_POLL_TIMEOUT_S", "180"))
    poll_interval_s = float(os.getenv("HISTORY_POLL_INTERVAL_S", "2"))
    deadline = time.time() + poll_timeout_s
    history = None

    while time.time() < deadline:
        try:
            history_payload = get_history(prompt_id)
            history = history_payload.get(prompt_id)

            if history is not None:
                outputs = history.get('outputs', {})
                if outputs:
                    break

        except Exception as e:
            logger.warning(f"History poll failed (prompt_id={prompt_id}): {e}")

        # If WS worked and we still do not see outputs yet, keep polling briefly for consistency.
        if not ws_failed and history is not None and history.get('outputs'):
            break
        time.sleep(poll_interval_s)

    if history is None:
        raise RuntimeError(f"Prompt history missing for prompt_id={prompt_id}")

    first_video = _extract_first_video_base64(history)
    if first_video:
        return first_video

    logger.error(f"No video outputs found. Available output nodes: {list(history.get('outputs', {}).keys())}")
    logger.error(f"Full history: {json.dumps(history, indent=2)}")
    status = history.get("status", {})
    messages = status.get("messages", [])
    for message in messages:
        if (
            isinstance(message, list)
            and len(message) >= 2
            and message[0] == "execution_error"
            and isinstance(message[1], dict)
        ):
            error_payload = message[1]
            exception_type = error_payload.get("exception_type") or "ExecutionError"
            node_type = error_payload.get("node_type") or "unknown_node"
            node_id = error_payload.get("node_id") or "unknown"
            exception_message = (error_payload.get("exception_message") or "No details").strip().replace("\n", " ")
            raise RuntimeError(
                f"{exception_type} at {node_type}({node_id}): {exception_message}"
            )

    raise RuntimeError("No video outputs produced")

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def apply_safe_attention_mode(prompt, requested_mode=None):
    """
    Guardrail: force non-Sage attention at runtime to avoid incompatible CUDA kernels
    on some RunPod GPU images.
    """
    mode = (
        str(requested_mode).strip().lower()
        if requested_mode is not None
        else os.getenv("ATTENTION_MODE_OVERRIDE", "sdpa").strip().lower()
    )
    safe_mode = "sdpa" if mode in ("", "sageattn", "auto") else mode
    changed_paths = []

    def _rewrite(obj, path):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                next_path = f"{path}.{key}" if path else str(key)
                if isinstance(value, str) and value.strip().lower() == "sageattn":
                    obj[key] = safe_mode
                    changed_paths.append(next_path)
                else:
                    _rewrite(value, next_path)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                _rewrite(value, f"{path}[{idx}]")

    _rewrite(prompt, "")

    if changed_paths:
        logger.warning(
            f"Replaced {len(changed_paths)} sageattn value(s) with {safe_mode}: "
            f"{', '.join(changed_paths[:10])}"
        )
    else:
        logger.info(f"No sageattn values found in prompt; attention mode remains {safe_mode}")

def handler(job):
        try:
            global COMFY_HTTP_READY
            job_input = job.get("input", {})

            logger.info(f"Received job input: {job_input}")
            task_id = f"task_{uuid.uuid4()}"

            # 이미지 입력 처리 (image_path, image_url, image_base64 중 하나만 사용)
            image_path = None
            if "image_path" in job_input:
                image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
            elif "image_url" in job_input:
                image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
            elif "image_base64" in job_input:
                image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
            else:
                # 기본값 사용
                image_path = "/example_image.png"
                logger.info("기본 이미지 파일을 사용합니다: /example_image.png")

            # 엔드 이미지 입력 처리 (end_image_path, end_image_url, end_image_base64 중 하나만 사용)
            end_image_path_local = None
            if "end_image_path" in job_input:
                end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
            elif "end_image_url" in job_input:
                end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
            elif "end_image_base64" in job_input:
                end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")

            # LoRA 설정 확인 - 배열로 받아서 처리
            lora_pairs = job_input.get("lora_pairs", [])

            # 최대 4개 LoRA까지 지원
            lora_count = min(len(lora_pairs), 4)
            if lora_count > len(lora_pairs):
                logger.warning(f"LoRA 개수가 {len(lora_pairs)}개입니다. 최대 4개까지만 지원됩니다. 처음 4개만 사용합니다.")
                lora_pairs = lora_pairs[:4]

            # 워크플로우 파일 선택 (end_image_*가 있으면 FLF2V 워크플로 사용)
            workflow_file = "/new_Wan22_flf2v_api.json" if end_image_path_local else "/new_Wan22_api.json"
            logger.info(f"Using {'FLF2V' if end_image_path_local else 'single'} workflow with {lora_count} LoRA pairs")

            prompt = load_workflow(workflow_file)
            apply_safe_attention_mode(prompt, job_input.get("attention_mode_override"))

            requested_length = _safe_int(job_input.get("length", DEFAULT_LENGTH), DEFAULT_LENGTH)
            requested_steps = _safe_int(job_input.get("steps", DEFAULT_STEPS), DEFAULT_STEPS)
            requested_context_frames = _safe_int(
                job_input.get("context_frames", min(requested_length, DEFAULT_CONTEXT_FRAMES)),
                min(requested_length, DEFAULT_CONTEXT_FRAMES),
            )
            requested_context_overlap = _safe_int(
                job_input.get("context_overlap", DEFAULT_CONTEXT_OVERLAP),
                DEFAULT_CONTEXT_OVERLAP,
            )

            if ENFORCE_SPEED_LIMITS:
                length = _clamp(requested_length, 16, MAX_LENGTH)
                steps = _clamp(requested_steps, 1, MAX_STEPS)
            else:
                length = max(16, requested_length)
                steps = max(1, requested_steps)

            context_frames = _clamp(requested_context_frames, 1, length)
            context_overlap_cap = min(MAX_CONTEXT_OVERLAP, max(0, context_frames - 1))
            context_overlap = _clamp(requested_context_overlap, 0, context_overlap_cap)

            if length != requested_length:
                logger.info(f"Length capped for speed: {requested_length} -> {length}")
            if steps != requested_steps:
                logger.info(f"Steps capped for speed: {requested_steps} -> {steps}")
            if context_overlap != requested_context_overlap:
                logger.info(
                    f"Context overlap adjusted: {requested_context_overlap} -> {context_overlap}"
                )

            seed = job_input.get("seed", 42)
            cfg = float(job_input.get("cfg", 2.0))
            original_width = job_input.get("width", 480)
            original_height = job_input.get("height", 832)

            prompt_text = job_input.get("prompt")
            if not prompt_text:
                return {"error": "Missing required field: prompt"}

            prompt["244"]["inputs"]["image"] = image_path
            prompt["541"]["inputs"]["num_frames"] = length
            prompt["135"]["inputs"]["positive_prompt"] = prompt_text
            prompt["135"]["inputs"]["negative_prompt"] = job_input.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
            prompt["220"]["inputs"]["seed"] = seed
            prompt["540"]["inputs"]["seed"] = seed
            prompt["540"]["inputs"]["cfg"] = cfg
            # 해상도(폭/높이) 16배수 보정
            adjusted_width = to_nearest_multiple_of_16(original_width)
            adjusted_height = to_nearest_multiple_of_16(original_height)
            if adjusted_width != original_width:
                logger.info(f"Width adjusted to nearest multiple of 16: {original_width} -> {adjusted_width}")
            if adjusted_height != original_height:
                logger.info(f"Height adjusted to nearest multiple of 16: {original_height} -> {adjusted_height}")
            prompt["235"]["inputs"]["value"] = adjusted_width
            prompt["236"]["inputs"]["value"] = adjusted_height
            prompt["498"]["inputs"]["context_overlap"] = context_overlap
            prompt["498"]["inputs"]["context_frames"] = context_frames

            # step 설정 적용
            if "834" in prompt:
                prompt["834"]["inputs"]["steps"] = steps
                logger.info(f"Steps set to: {steps}")
                lowsteps = max(1, int(round(steps * 0.6)))
                prompt["829"]["inputs"]["step"] = lowsteps
                logger.info(f"LowSteps set to: {lowsteps}")

            # 엔드 이미지가 있는 경우 617번 노드에 경로 적용 (FLF2V 전용)
            if end_image_path_local:
                prompt["617"]["inputs"]["image"] = end_image_path_local

            # LoRA 설정 적용 - HIGH LoRA는 노드 279, LOW LoRA는 노드 553
            if lora_count > 0:
                # HIGH LoRA 노드 (279번)
                high_lora_node_id = "279"

                # LOW LoRA 노드 (553번)
                low_lora_node_id = "553"

                # 입력받은 LoRA pairs 적용 (lora_1부터 시작)
                for i, lora_pair in enumerate(lora_pairs):
                    if i < 4:  # 최대 4개까지만
                        lora_high = lora_pair.get("high")
                        lora_low = lora_pair.get("low")
                        lora_high_weight = lora_pair.get("high_weight", 1.0)
                        lora_low_weight = lora_pair.get("low_weight", 1.0)

                        # HIGH LoRA 설정 (노드 279번, lora_1부터 시작)
                        if lora_high:
                            prompt[high_lora_node_id]["inputs"][f"lora_{i+1}"] = lora_high
                            prompt[high_lora_node_id]["inputs"][f"strength_{i+1}"] = lora_high_weight
                            logger.info(f"LoRA {i+1} HIGH applied to node 279: {lora_high} with weight {lora_high_weight}")

                        # LOW LoRA 설정 (노드 553번, lora_1부터 시작)
                        if lora_low:
                            prompt[low_lora_node_id]["inputs"][f"lora_{i+1}"] = lora_low
                            prompt[low_lora_node_id]["inputs"][f"strength_{i+1}"] = lora_low_weight
                            logger.info(f"LoRA {i+1} LOW applied to node 553: {lora_low} with weight {lora_low_weight}")

            ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
            logger.info(f"Connecting to WebSocket: {ws_url}")

            # HTTP readiness check once per worker process.
            if not COMFY_HTTP_READY:
                http_url = f"http://{server_address}:8188/"
                logger.info(f"Checking HTTP connection to: {http_url}")
                max_http_attempts = int(os.getenv("HTTP_CONNECT_MAX_ATTEMPTS", "60"))
                http_retry_sleep_s = float(os.getenv("HTTP_CONNECT_RETRY_S", "1"))
                for http_attempt in range(max_http_attempts):
                    try:
                        urllib.request.urlopen(http_url, timeout=5)
                        logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
                        COMFY_HTTP_READY = True
                        break
                    except Exception as e:
                        logger.warning(
                            f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}"
                        )
                        if http_attempt == max_http_attempts - 1:
                            raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
                        time.sleep(http_retry_sleep_s)

            ws = websocket.WebSocket()
            # 웹소켓 연결 시도
            ws_retry_sleep_s = float(os.getenv("WS_CONNECT_RETRY_S", "2"))
            max_attempts = int(os.getenv("WS_CONNECT_MAX_ATTEMPTS", "60"))
            for attempt in range(max_attempts):
                try:
                    ws.connect(ws_url)
                    logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
                    break
                except Exception as e:
                    logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
                    if attempt == max_attempts - 1:
                        raise Exception("웹소켓 연결 시간 초과")
                    time.sleep(ws_retry_sleep_s)
            video_data = get_video_base64(ws, prompt)
            ws.close()
            return {"video": video_data}
        except Exception as e:
            logger.exception("Handler crashed")
            return {"error": f"{type(e).__name__}: {str(e)}"}

runpod.serverless.start({"handler": handler})
