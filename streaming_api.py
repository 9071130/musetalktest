"""
MuseTalk 流式视频输出 Web API
支持实时生成虚拟人视频并流式推送给前端
使用 FastAPI 框架
"""

import os
import sys
import cv2
import torch
import argparse
import threading
import queue
import time
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from omegaconf import OmegaConf
from transformers import WhisperModel

# 导入 MuseTalk 相关模块
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from scripts.realtime_inference import Avatar, fast_check_ffmpeg

# 创建 FastAPI 应用
app = FastAPI(
    title="MuseTalk Streaming API",
    description="实时虚拟人视频生成流式输出 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
device = None
vae = None
unet = None
pe = None
whisper = None
audio_processor = None
fp = None
weight_dtype = None
timesteps = None
args = None

# 流式输出相关
frame_queues = {}  # {session_id: queue.Queue()} 存储每个会话的帧队列
active_sessions = {}  # {session_id: {'avatar': Avatar, 'status': 'running'/'idle'}}
session_lock = threading.Lock()


# Pydantic 模型定义
class CreateAvatarRequest(BaseModel):
    avatar_id: str = "default"
    video_path: str
    bbox_shift: int = 0
    batch_size: int = 20


class StartInferenceRequest(BaseModel):
    avatar_id: str = "default"
    audio_path: str
    fps: int = 25
    skip_save_images: bool = True


class AvatarInfo(BaseModel):
    avatar_id: str
    status: str
    created_at: float


class AvatarListResponse(BaseModel):
    avatars: List[AvatarInfo]


def generate_frames(session_id: str):
    """生成 MJPEG 视频流"""
    while True:
        if session_id not in frame_queues:
            break
        
        try:
            # 从队列获取帧（最多等待1秒）
            frame = frame_queues[session_id].get(timeout=1)
            
            # 将 BGR 转换为 JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # 生成 MJPEG 格式的响应
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # 超时，发送心跳帧（可选）
            continue
        except Exception as e:
            print(f"Error generating frame for session {session_id}: {e}")
            break


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_file = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>MuseTalk Streaming API</h1><p>请确保 templates/index.html 文件存在</p>")


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "message": "MuseTalk Streaming API is running"}


@app.post("/api/avatar/create")
async def create_avatar(request: CreateAvatarRequest):
    """创建 Avatar（预处理）"""
    try:
        if not request.video_path or not os.path.exists(request.video_path):
            raise HTTPException(status_code=400, detail="video_path is required and must exist")
        
        # 创建 Avatar 实例（preparation=True 会进行预处理）
        avatar = Avatar(
            avatar_id=request.avatar_id,
            video_path=request.video_path,
            bbox_shift=request.bbox_shift,
            batch_size=request.batch_size,
            preparation=True
        )
        
        with session_lock:
            active_sessions[request.avatar_id] = {
                'avatar': avatar,
                'status': 'idle',
                'created_at': time.time()
            }
        
        return {
            "status": "success",
            "message": f"Avatar {request.avatar_id} created successfully",
            "avatar_id": request.avatar_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/avatar/list", response_model=AvatarListResponse)
async def list_avatars():
    """列出所有已创建的 Avatar"""
    with session_lock:
        avatar_list = []
        for avatar_id, session_info in active_sessions.items():
            avatar_list.append(AvatarInfo(
                avatar_id=avatar_id,
                status=session_info['status'],
                created_at=session_info['created_at']
            ))
    
    return AvatarListResponse(avatars=avatar_list)


@app.post("/api/inference/start")
async def start_inference(request: StartInferenceRequest, background_tasks: BackgroundTasks):
    """开始推理并流式输出"""
    try:
        if not request.audio_path or not os.path.exists(request.audio_path):
            raise HTTPException(status_code=400, detail="audio_path is required and must exist")
        
        if request.avatar_id not in active_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Avatar {request.avatar_id} not found. Please create it first."
            )
        
        # 创建会话 ID
        session_id = f"{request.avatar_id}_{int(time.time())}"
        
        # 创建帧队列
        frame_queues[session_id] = queue.Queue(maxsize=30)  # 限制队列大小，避免内存溢出
        
        # 定义帧回调函数
        def frame_callback(frame, frame_idx):
            """每生成一帧就放入队列"""
            try:
                # 如果队列满了，丢弃最旧的帧
                if frame_queues[session_id].full():
                    try:
                        frame_queues[session_id].get_nowait()
                    except queue.Empty:
                        pass
                frame_queues[session_id].put(frame.copy())
            except Exception as e:
                print(f"Error in frame callback: {e}")
        
        # 更新会话状态
        with session_lock:
            active_sessions[request.avatar_id]['status'] = 'running'
        
        # 在后台任务中运行推理
        def run_inference():
            try:
                avatar = active_sessions[request.avatar_id]['avatar']
                avatar.inference(
                    audio_path=request.audio_path,
                    out_vid_name=None,  # 不保存视频文件
                    fps=request.fps,
                    skip_save_images=request.skip_save_images,
                    frame_callback=frame_callback
                )
            except Exception as e:
                print(f"Error in inference: {e}")
            finally:
                # 推理完成后，发送结束标记
                with session_lock:
                    active_sessions[request.avatar_id]['status'] = 'idle'
                # 等待一段时间后清理队列
                time.sleep(2)
                if session_id in frame_queues:
                    del frame_queues[session_id]
        
        # 添加后台任务
        background_tasks.add_task(run_inference)
        
        return {
            "status": "success",
            "message": "Inference started",
            "session_id": session_id,
            "stream_url": f"/api/stream/{session_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/{session_id}")
async def video_stream(session_id: str):
    """MJPEG 视频流端点"""
    if session_id not in frame_queues:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return StreamingResponse(
        generate_frames(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/inference/stop/{session_id}")
async def stop_inference(session_id: str):
    """停止推理并清理会话"""
    if session_id in frame_queues:
        del frame_queues[session_id]
        return {"status": "success", "message": f"Session {session_id} stopped"}
    raise HTTPException(status_code=404, detail="Session not found")


def init_models():
    """初始化模型"""
    global device, vae, unet, pe, whisper, audio_processor, fp, weight_dtype, timesteps
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print("Loading models...")
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    # 转换为半精度
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    weight_dtype = unet.model.dtype
    
    # 初始化音频处理器和 Whisper
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # 初始化人脸解析器
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    
    # 设置 realtime_inference 模块的全局变量
    import scripts.realtime_inference as rt_module
    rt_module.args = args
    rt_module.device = device
    rt_module.vae = vae
    rt_module.unet = unet
    rt_module.pe = pe
    rt_module.whisper = whisper
    rt_module.audio_processor = audio_processor
    rt_module.weight_dtype = weight_dtype
    rt_module.timesteps = timesteps
    rt_module.fp = fp
    
    print("Models loaded successfully!")


if __name__ == '__main__':
    import uvicorn
    
    parser = argparse.ArgumentParser(description='MuseTalk Streaming API Server (FastAPI)')
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 (already default)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    
    args = parser.parse_args()
    
    # 配置 ffmpeg 路径
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    
    # 初始化模型（会自动设置 realtime_inference 模块的全局变量）
    init_models()
    
    print(f"Starting MuseTalk Streaming API server on http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Alternative docs: http://{args.host}:{args.port}/redoc")
    
    # 使用 uvicorn 运行 FastAPI 应用
    uvicorn.run(
        "streaming_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
