# MuseTalk 流式视频输出 API 使用说明

## 概述

本项目已添加流式视频输出功能，支持实时生成虚拟人视频并推送给前端 Web 页面。主要特性：

- ✅ **流式输出**：边生成边推送，无需等待全部完成
- ✅ **Web API**：RESTful API 接口，易于集成
- ✅ **MJPEG 流**：浏览器原生支持，无需额外插件
- ✅ **Avatar 复用**：一次预处理，多次使用

## 安装依赖

```bash
pip install -r requirements.txt
```

新增依赖：
- `fastapi` - 现代、快速的 Web 框架
- `uvicorn[standard]` - ASGI 服务器
- `python-multipart` - 文件上传支持

## 启动服务

```bash
python streaming_api.py --host 0.0.0.0 --port 5000
```

### 参数说明

- `--host`: 服务器地址（默认: 0.0.0.0）
- `--port`: 端口号（默认: 5000）
- `--reload`: 开发模式，自动重载（仅开发环境使用）
- `--gpu_id`: GPU ID（默认: 0）
- `--unet_model_path`: UNet 模型路径（默认: ./models/musetalkV15/unet.pth）
- `--whisper_dir`: Whisper 模型目录（默认: ./models/whisper）
- 其他参数与 `realtime_inference.py` 相同

### API 文档

FastAPI 自动生成交互式 API 文档：

- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

你可以在浏览器中直接测试所有 API 接口！

## API 接口

### 1. 健康检查

```http
GET /api/health
```

响应：
```json
{
  "status": "ok",
  "message": "MuseTalk Streaming API is running"
}
```

### 2. 创建 Avatar

```http
POST /api/avatar/create
Content-Type: application/json

{
  "avatar_id": "avatar1",
  "video_path": "data/video/yongen.mp4",
  "bbox_shift": 0,
  "batch_size": 20
}
```

响应：
```json
{
  "status": "success",
  "message": "Avatar avatar1 created successfully",
  "avatar_id": "avatar1"
}
```

### 3. 列出所有 Avatar

```http
GET /api/avatar/list
```

响应：
```json
{
  "avatars": [
    {
      "avatar_id": "avatar1",
      "status": "idle",
      "created_at": 1234567890.123
    }
  ]
}
```

### 4. 开始推理（流式输出）

```http
POST /api/inference/start
Content-Type: application/json

{
  "avatar_id": "avatar1",
  "audio_path": "data/audio/yongen.wav",
  "fps": 25,
  "skip_save_images": true
}
```

响应：
```json
{
  "status": "success",
  "message": "Inference started",
  "session_id": "avatar1_1234567890",
  "stream_url": "/api/stream/avatar1_1234567890"
}
```

### 5. 获取视频流（MJPEG）

```http
GET /api/stream/{session_id}
```

这是一个 MJPEG 流端点，可以直接在 HTML 中使用：

```html
<img src="http://localhost:5000/api/stream/avatar1_1234567890">
```

### 6. 停止推理

```http
POST /api/inference/stop/{session_id}
```

响应：
```json
{
  "status": "success",
  "message": "Session avatar1_1234567890 stopped"
}
```

## Web 界面使用

启动服务后，访问 `http://localhost:5000` 即可使用内置的 Web 界面。

### 使用步骤

1. **创建 Avatar**
   - 输入 Avatar ID（例如：avatar1）
   - 输入视频文件路径（例如：data/video/yongen.mp4）
   - 点击"创建 Avatar"按钮
   - 等待预处理完成（可能需要几分钟）

2. **查看 Avatar 列表**
   - 点击"刷新列表"查看已创建的 Avatar

3. **开始生成视频**
   - 选择或输入 Avatar ID
   - 输入音频文件路径（例如：data/audio/yongen.wav）
   - 点击"开始生成"按钮
   - 视频流将自动显示在页面上

4. **停止生成**
   - 点击"停止生成"按钮

## 前端集成示例

### JavaScript 示例

```javascript
// 1. 创建 Avatar
async function createAvatar() {
  const response = await fetch('http://localhost:5000/api/avatar/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      avatar_id: 'avatar1',
      video_path: 'data/video/yongen.mp4',
      bbox_shift: 0,
      batch_size: 20
    })
  });
  const data = await response.json();
  console.log(data);
}

// 2. 开始推理
async function startInference() {
  const response = await fetch('http://localhost:5000/api/inference/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      avatar_id: 'avatar1',
      audio_path: 'data/audio/yongen.wav',
      fps: 25,
      skip_save_images: true
    })
  });
  const data = await response.json();
  
  // 3. 显示视频流
  const videoImg = document.getElementById('videoStream');
  videoImg.src = `http://localhost:5000${data.stream_url}`;
}
```

### HTML 示例

```html
<!DOCTYPE html>
<html>
<head>
  <title>MuseTalk 视频流</title>
</head>
<body>
  <h1>实时视频流</h1>
  <img id="videoStream" src="" alt="视频流">
  
  <script>
    // 启动推理后，设置视频流 URL
    const sessionId = 'avatar1_1234567890';
    document.getElementById('videoStream').src = 
      `http://localhost:5000/api/stream/${sessionId}`;
  </script>
</body>
</html>
```

## 技术架构

### 流式输出流程

1. **预处理阶段**（创建 Avatar 时）
   - 视频拆帧
   - 人脸检测和裁剪
   - VAE 编码 latent
   - 生成 mask
   - 所有数据保存到磁盘

2. **推理阶段**（开始推理时）
   - 音频特征提取（Whisper）
   - 批量生成嘴型帧（UNet + VAE）
   - 每生成一帧立即回调
   - 回调函数将帧放入队列

3. **流式输出阶段**
   - 后台线程从队列取帧
   - 编码为 JPEG
   - 通过 MJPEG 流推送给前端
   - 前端自动刷新显示

### 关键代码修改

1. **`scripts/realtime_inference.py`**
   - `process_frames()` 方法添加 `frame_callback` 参数
   - `inference()` 方法添加 `frame_callback` 参数
   - 每生成一帧立即调用回调函数

2. **`streaming_api.py`**
   - Flask Web 服务器
   - Avatar 管理 API
   - 推理控制 API
   - MJPEG 流端点

## 注意事项

1. **首次创建 Avatar 需要时间**：预处理过程可能需要几分钟，请耐心等待

2. **内存管理**：帧队列有大小限制（30帧），避免内存溢出

3. **并发处理**：当前版本支持多个会话，但建议不要同时运行太多推理任务

4. **文件路径**：确保视频和音频文件路径正确，支持相对路径和绝对路径

5. **GPU 内存**：确保 GPU 有足够内存，建议至少 8GB

## 故障排查

### 问题：视频流不显示

- 检查浏览器控制台是否有错误
- 确认 `session_id` 是否正确
- 检查服务器日志是否有错误信息

### 问题：Avatar 创建失败

- 检查视频文件是否存在
- 确认视频文件格式正确（支持 mp4, avi 等）
- 查看服务器日志获取详细错误信息

### 问题：推理启动失败

- 确认 Avatar 已成功创建
- 检查音频文件是否存在
- 确认音频文件格式正确（支持 wav, mp3 等）

## 性能优化建议

1. **使用 `skip_save_images=True`**：如果只需要流式输出，不需要保存 PNG 文件

2. **调整 `batch_size`**：根据 GPU 内存调整，越大越快但占用内存更多

3. **使用 float16**：默认已启用，可进一步减少内存占用

4. **预处理一次，多次使用**：Avatar 创建后可以反复使用，无需重新预处理

## 后续扩展

可以基于当前架构扩展以下功能：

- WebSocket 支持（更低延迟）
- 音频流输入（实时语音驱动）
- 多 Avatar 切换
- 视频录制功能
- 参数实时调整

## 许可证

与 MuseTalk 主项目相同（MIT License）

