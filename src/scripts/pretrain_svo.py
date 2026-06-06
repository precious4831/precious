"""
pretrain_svo.py -- Stage 1: SVO RSSM Categorical 预训练

========================================================================
训练流程:
  Step 0: python data_collector.py --episodes 300 --output svo_dataset.npz
  Step 1: python pretrain_svo.py --dataset svo_dataset.npz --output svo_pretrained.pt
  Step 2: python train.py --stage l1
  Step 3: python train.py --stage l2 --svo_pretrained svo_pretrained.pt ...
========================================================================

相比旧版的核心变化:
  旧版: 每个样本单步 compute_elbo (h=None, 独立样本)
  新版: 每个样本 10步BPTT展开 forward_sequence
        每步算 Categorical KL(pi_post || pi_prior)
        末步算 Recon + Anchor
        KL逼迫先验追上后验，推理时先验可独立工作

监控指标:
  theta_post : 后验加权SVO角度（训练时看到轨迹）
  theta_prior: 先验加权SVO角度（推理时用）
  conf       : pi_max均值（意图确信度，趋近1=训练成功）
  目标       : theta_post ≈ theta_prior，conf 趋近于高值

使用方法:
  python pretrain_svo.py --dataset svo_dataset.npz --output svo_pretrained.pt
  python pretrain_svo.py --dataset svo_dataset.npz --epochs 150 --anchor_weight 3.0
"""

from pathlib import Path
import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config
from src.models.svo_model import SVOVariationalBIRL, NUM_INTENTS, INTENT_ANCHORS_DEG


# ======================================================================== #
#  数据集                                                                    #
# ======================================================================== #

class SVOTrajectoryDataset(Dataset):
    """
    SVO预训练数据集。

    每个样本: (ego_past, npc_past, npc_future, target_mu)
      ego_past  : (T=10, 5)  [x, y, yaw, vx, vy]
      npc_past  : (T=10, 5)
      npc_future: (H=10, 5)
      target_mu : scalar     弱监督锚点角度（来自风格标签）

    过滤: 距离 < max_dist, NPC均速 > min_npc_speed, 速度变化 > min_speed_var
    """

    STYLE_TO_MU = {
        'aggressive':        15.0,
        'semi_aggressive':   30.0,
        'normal':            45.0,
        'semi_conservative': 60.0,
        'conservative':      75.0,
    }

    def __init__(self, npz_path, max_dist=50.0, min_npc_speed=0.5, min_speed_var=0.1):
        data       = np.load(npz_path, allow_pickle=True)
        ego_past   = data['ego_past'].astype(np.float32)
        npc_past   = data['npc_past'].astype(np.float32)
        npc_future = data['npc_future'].astype(np.float32)
        raw_styles = (data['styles']
                      if 'styles' in data
                      else np.array(['normal'] * len(ego_past)))
        N_raw = len(ego_past)
        print(f"原始数据: {N_raw} 样本")

        dx = ego_past[:, :, 0] - npc_past[:, :, 0]
        dy = ego_past[:, :, 1] - npc_past[:, :, 1]
        mask_dist  = np.min(np.sqrt(dx**2 + dy**2), axis=1) < max_dist
        npc_speed  = np.sqrt(npc_past[:, :, 3]**2 + npc_past[:, :, 4]**2)
        mask_speed = npc_speed.mean(axis=1) > min_npc_speed
        mask_var   = npc_speed.std(axis=1) > min_speed_var
        valid = mask_dist & mask_speed & mask_var

        self.ego_past   = ego_past[valid]
        self.npc_past   = npc_past[valid]
        self.npc_future = npc_future[valid]
        self.styles     = raw_styles[valid]
        self.target_mus = np.array(
            [self.STYLE_TO_MU.get(str(s), 45.0) for s in self.styles],
            dtype=np.float32,
        )

        print(f"筛选后: {valid.sum()} / {N_raw} 样本")
        print(f"  dist<{max_dist}m: {mask_dist.sum()}, "
              f"speed>{min_npc_speed}: {mask_speed.sum()}, "
              f"var>{min_speed_var}: {mask_var.sum()}")

        from collections import Counter
        print(f"  风格分布: {dict(Counter(str(s) for s in self.styles))}")

    def __len__(self):
        return len(self.ego_past)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.ego_past[idx]),
            torch.from_numpy(self.npc_past[idx]),
            torch.from_numpy(self.npc_future[idx]),
            torch.tensor(self.target_mus[idx]),
        )


# ======================================================================== #
#  Beta 调度                                                                 #
# ======================================================================== #

def get_beta(epoch, warmup_epochs=20, max_beta=0.1, delay=10):
    """
    KL权重退火: 前 delay epochs=0，然后线性增加到 max_beta。

    Categorical KL 上界 = log(5) ≈ 1.6（均匀→某类的最大KL）
    max_beta=0.1: recon(~1.0) + 0.1*KL(0~1.6) + anchor → 三者均衡
    前 delay=10 epoch KL=0，让 recon 和 anchor 先收敛，再引入KL约束
    """
    if epoch < delay:
        return 0.0
    progress = (epoch - delay) / max(warmup_epochs, 1)
    return min(max_beta, max_beta * progress)


# ======================================================================== #
#  训练 / 验证                                                               #
# ======================================================================== #

def train_epoch(model, dataloader, optimizer, device, beta):
    """
    训练一个 epoch。
    调用 model.forward_sequence()：TrajEncoder 处理完整窗口，RSSM 单步更新。
    """
    model.train()
    total_loss    = 0.0
    total_metrics = {}
    n_batches     = 0

    for ego_past, npc_past, npc_future, target_mu in dataloader:
        ego_past   = ego_past.to(device)
        npc_past   = npc_past.to(device)
        npc_future = npc_future.to(device)
        target_mu  = target_mu.to(device)

        loss, metrics = model.forward_sequence(
            npc_past, ego_past, npc_future,
            target_mu=target_mu, beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    avg          = {k: v / n_batches for k, v in total_metrics.items()}
    avg['loss']  = total_loss / n_batches
    return avg


@torch.no_grad()
def validate(model, dataloader, device, beta):
    """验证一个 epoch。"""
    model.eval()
    total_loss    = 0.0
    total_metrics = {}
    n_batches     = 0

    for ego_past, npc_past, npc_future, target_mu in dataloader:
        ego_past   = ego_past.to(device)
        npc_past   = npc_past.to(device)
        npc_future = npc_future.to(device)
        target_mu  = target_mu.to(device)

        loss, metrics = model.forward_sequence(
            npc_past, ego_past, npc_future,
            target_mu=target_mu, beta=beta,
        )

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    avg         = {k: v / n_batches for k, v in total_metrics.items()}
    avg['loss'] = total_loss / n_batches
    return avg


# ======================================================================== #
#  参数解析                                                                  #
# ======================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description='SVO RSSM Categorical 预训练')

    parser.add_argument('--dataset',    type=str,   default=r'D:\桌面\毕设代码\SVO-CVaR\pretrain_svo\svo_dataset 3.11 300ep.npz')
    parser.add_argument('--output',     type=str,   default='svo_pretrained.pt')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--val_ratio',  type=float, default=0.1)

    # Beta 退火
    parser.add_argument('--max_beta',
                        type=float, default=0.1,
                        help='KL权重上限 (Categorical KL上界≈1.6，0.1合理)')
    parser.add_argument('--beta_warmup_epochs', type=int, default=20)
    parser.add_argument('--beta_delay',
                        type=int, default=10,
                        help='前N个epoch KL=0，先让recon收敛')

    # Anchor 权重
    parser.add_argument('--anchor_weight',
                        type=float, default=None,
                        help='弱监督锚定权重 (None=用config默认值)')

    parser.add_argument('--device',  type=str, default='auto')
    parser.add_argument('--seed',    type=int, default=38)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard日志目录')

    return parser.parse_args()


# ======================================================================== #
#  主函数                                                                    #
# ======================================================================== #

def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # === 配置 ===
    config = get_default_config()
    if args.anchor_weight is not None:
        config.svo.anchor_loss_weight = args.anchor_weight
    print(f"Anchor权重: {config.svo.anchor_loss_weight}")
    print(f"意图锚点: {INTENT_ANCHORS_DEG}°  (固定，不可学习)")

    # === 数据集 ===
    dataset = SVOTrajectoryDataset(args.dataset)
    n_val   = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"训练集: {n_train}，验证集: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    # === 模型 ===
    model    = SVOVariationalBIRL(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    for name, mod in [('traj_encoder', model.traj_encoder),
                      ('rssm_cell',    model.rssm_cell),
                      ('decoder',      model.decoder)]:
        print(f"  {name}: {sum(p.numel() for p in mod.parameters()):,}")

    # === 优化器 ===
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = args.log_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.output)), 'svo_pretrain_logs'
        )
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")
    except ImportError:
        print("TensorBoard不可用")

    # === 训练循环 ===
    print()
    print("=" * 65)
    print("开始SVO RSSM预训练 (10步BPTT，Categorical意图分布)")
    print(f"Epochs={args.epochs} | LR={args.lr} | Batch={args.batch_size}")
    print(f"Beta: delay={args.beta_delay}ep, "
          f"warmup={args.beta_warmup_epochs}ep, max={args.max_beta}")
    print(f"Anchor weight: {config.svo.anchor_loss_weight}")
    print()
    print("监控指标说明:")
    print("  theta_post : 后验加权SVO角度（训练时看到轨迹）")
    print("  theta_prior: 先验加权SVO角度（推理时用这个）")
    print("  conf       : pi_max均值（意图确信度，趋近1=训练成功）")
    print("  目标       : theta_post ≈ theta_prior，conf 趋近于高值")
    print("=" * 65)

    best_val_loss = float('inf')
    start_time    = time.time()

    for epoch in range(args.epochs):
        beta    = get_beta(epoch, args.beta_warmup_epochs, args.max_beta, args.beta_delay)
        train_m = train_epoch(model, train_loader, optimizer, device, beta)
        val_m   = validate(model, val_loader, device, beta)

        scheduler.step(val_m['loss'])
        lr = optimizer.param_groups[0]['lr']

        # 日志
        log_line = (
            f"Epoch {epoch+1:3d}/{args.epochs} | beta={beta:.3f} | "
            f"train={train_m['loss']:.4f} "
            f"rec={train_m['recon_loss']:.4f} "
            f"kl={train_m['kl_loss']:.4f} "
            f"anc={train_m['anchor_loss']:.4f} | "
            f"val={val_m['loss']:.4f} | "
            f"theta_post={val_m['theta_mean']:.1f}deg "
            f"theta_prior={val_m.get('theta_prior_mean', 0.0):.1f}deg "
            f"conf={val_m['pi_max_mean']:.3f} | "
            f"lr={lr:.2e}"
        )
        print(log_line)

        if writer:
            for tag, val in [
                ('pretrain/train_loss',  train_m['loss']),
                ('pretrain/val_loss',    val_m['loss']),
                ('pretrain/recon_loss',  train_m['recon_loss']),
                ('pretrain/kl_loss',     train_m['kl_loss']),
                ('pretrain/anchor_loss', train_m['anchor_loss']),
                ('pretrain/theta_post',  val_m['theta_mean']),
                ('pretrain/theta_prior', val_m.get('theta_prior_mean', 0.0)),
                ('pretrain/conf_post',   val_m['pi_max_mean']),
                ('pretrain/conf_prior',  val_m.get('pi_prior_max_mean', 0.0)),
                ('pretrain/beta',        beta),
                ('pretrain/lr',          lr),
            ]:
                writer.add_scalar(tag, val, epoch)

        # 保存最佳
        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            torch.save({
                'svo_birl':  model.state_dict(),
                'epoch':     epoch + 1,
                'val_loss':  best_val_loss,
                'config': {
                    'hidden_dim':         config.svo.hidden_dim,
                    'prediction_horizon': config.svo.prediction_horizon,
                    'anchor_loss_weight': config.svo.anchor_loss_weight,
                    'num_intents':        NUM_INTENTS,
                    'intent_anchors_deg': INTENT_ANCHORS_DEG,
                },
            }, args.output)
            print(f"  -> 新最佳! val={best_val_loss:.4f} "
                  f"theta_post={val_m['theta_mean']:.1f}deg "
                  f"theta_prior={val_m.get('theta_prior_mean', 0.0):.1f}deg "
                  f"conf={val_m['pi_max_mean']:.3f}")

    elapsed = time.time() - start_time
    print()
    print("=" * 65)
    print("SVO RSSM预训练完成!")
    print(f"用时: {elapsed / 60:.1f} 分钟 | 最佳val_loss: {best_val_loss:.4f}")
    print(f"模型保存至: {args.output}")
    print()
    print("下一步: python train.py --stage l2 --svo_pretrained", args.output)
    print("=" * 65)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
