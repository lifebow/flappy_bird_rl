# Flappy Bird Reinforcement Learning (Parallel Training)

Dự án này sử dụng thuật toán **PPO (Proximal Policy Optimization)** từ thư viện **Stable Baselines3** để huấn luyện một Agent chơi game Flappy Bird. Dự án hỗ trợ huấn luyện song song đa môi trường, tận dụng GPU, và cho phép fine-tune model.

## 🚀 Tính năng chính

- **Huấn luyện song song**: Sử dụng `SubprocVecEnv` để chạy 12 môi trường cùng lúc, tăng tốc độ học gấp nhiều lần.
- **Tốc độ cao (Speed-up)**: Hỗ trợ chế độ không giới hạn FPS trong lúc training (đạt ~3000+ FPS trên RTX 3090).
- **GPU Accelerated**: Tự động nhận diện và sử dụng CUDA để tính toán mạng neural.
- **Rank-0 Rendering**: Chỉ hiển thị duy nhất 1 màn hình game của instance chính để giám sát.
- **Fine-tuning**: Tự động tải lại model cũ để tiếp tục huấn luyện nếu có.
- **Auto-Save Best Model**: Tự động lưu model tốt nhất mỗi khi đạt điểm cao mới.
- **Periodic Checkpoints**: Lưu checkpoint định kỳ mỗi 50,000 steps vào thư mục `checkpoints/`.
- **Nuanced Reward**: Hệ thống phần thưởng thông minh, giảm mức phạt khi chim va chạm gần tâm khe hở.

## 🛠 Cấu trúc thư mục

- `config.py`: Các hằng số vật lý (trọng lực, lực nhảy) và cấu hình môi trường.
- `game.py`: Logic cốt lõi của game (vật lý, va chạm, render điểm số).
- `env.py`: Wrapper theo chuẩn **Gymnasium** để kết nối Game với RL.
- `train.py`: Script huấn luyện chính (PPO, Callbacks, Parallel Env).
- `play.py`: Chế độ dành cho người chơi (Dùng phím **Space**).
- `eval.py`: Chế độ xem Agent đã học được (tải model từ file `.zip`).
- `checkpoints/`: Thư mục chứa các checkpoint định kỳ.
- `best_model.zip`: Model tốt nhất được lưu tự động.

## 📦 Cài đặt

Yêu cầu Python 3.8+. Nên cài đặt trong môi trường ảo.

```bash
pip install gymnasium stable-baselines3 pygame shimmy torch
# Nếu dùng GPU NVIDIA, hãy cài đặt torch kèm CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🎮 Cách sử dụng

### 1. Huấn luyện Agent

Mặc định sẽ chạy 12 CPU song song và sử dụng GPU nếu có.

```bash
python train.py
```

**Outputs**:

- `best_model.zip`: Tự động lưu khi đạt điểm cao mới
- `checkpoints/ppo_checkpoint_*.zip`: Checkpoints mỗi 50k steps
- `ppo_flappy_bird_new.zip`: Model cuối cùng

*Nhấn Ctrl+C để dừng huấn luyện bất cứ lúc nào.*

### 2. Tự trải nghiệm (Manual Play)

Tự tay điều khiển chim để cảm nhận độ khó của game.

```bash
python play.py
```

### 3. Đánh giá Agent (Evaluation)

Xem "thành quả" của Agent sau khi huấn luyện.

```bash
python eval.py
```

## 📝 Reward Logic

Hệ thống Reward được thiết kế để khuyến khích chim bay qua khe hở:

- `+0.1` cho mỗi frame còn sống.
- `+1.0` khi vượt qua ống.
- `Penalty = -1.0 * (distance_to_center / max_distance)` khi va chạm. (Va chạm gần tâm khe hở sẽ bị trừ ít điểm hơn va chạm xa).
