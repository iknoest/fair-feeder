# Movable Monitor — Equipment Shopping List

## Already Owned

| Item | Notes |
|------|-------|
| Raspberry Pi 5 | Main brain — runs SLAM, YOLO, Telegram bot |
| Arduino | Motor control, fork lift servo, sensor reading |
| Bambu Lab P1S (planned) | 3D print chassis and fork mechanism |

---

## To Buy — Sensors & Camera

| # | Item (EN) | 淘寶搜尋關鍵字 | Est. Price | Notes |
|---|-----------|---------------|-----------|-------|
| 1 | **OAK-D Lite** depth camera | OAK-D Lite 深度相機 / OAK-D Lite 双目摄像头 | ~$99 / ¥700 | Stereo depth + RGB + AI chip. Main camera for SLAM + cat detection |
| 2 | **IR LED ring** (850nm) | 850nm 红外补光灯环 / 红外LED灯板 | ~$5 / ¥15-35 | Night vision illumination. Match OAK-D Lite lens size. 850nm is invisible to cats |
| 3 | **IR pass filter** (optional) | 红外滤光片 850nm | ~$3 / ¥10-20 | Improves IR image quality at night. May not be needed — test first |

---

## To Buy — Motors & Drive

| # | Item (EN) | 淘寶搜尋關鍵字 | Est. Price | Notes |
|---|-----------|---------------|-----------|-------|
| 4 | **DC gear motor with encoder** x2 | JGB37-520 编码器减速电机 / 带编码器直流减速电机 | ~$8-15 ea / ¥25-50 | Wheel drive motors. Encoder needed for odometry (SLAM). 12V, ~200RPM |
| 5 | **Motor driver board** | L298N 电机驱动板 / TB6612FNG 电机驱动模块 | ~$3-5 / ¥10-20 | TB6612FNG recommended (more efficient, less heat than L298N) |
| 6 | **Servo motor** (for fork lift) | MG996R 舵机 / 大扭力舵机 | ~$5-8 / ¥15-30 | Lifts the fork/camera platform. MG996R has enough torque. Or use 2x SG90 for lighter load |
| 7 | **Caster wheel** x1 | 万向轮 / 脚轮 1寸 | ~$2 / ¥5-10 | Rear support wheel (2 drive wheels front + 1 caster back = differential drive) |
| 8 | **Rubber wheels** (65-80mm) x2 | 65mm 橡胶轮 / 智能小车轮子 带联轴器 | ~$3-5 / ¥10-20 | Match motor shaft diameter. Rubber for grip on floor |

---

## To Buy — Power

| # | Item (EN) | 淘寶搜尋關鍵字 | Est. Price | Notes |
|---|-----------|---------------|-----------|-------|
| 9 | **LiPo battery** 11.1V 3S 2200mAh | 3S 11.1V 2200mAh 锂电池 | ~$15-20 / ¥50-80 | Powers everything. 3S gives 12V for motors + 5V via step-down for Pi 5 |
| 10 | **Buck converter** (12V→5V 5A) | DC-DC 降压模块 12V转5V 5A | ~$3 / ¥8-15 | Powers Pi 5 (needs 5V/5A). Get one with USB-C output if possible |
| 11 | **Battery charger** (3S balance) | 3S 平衡充电器 / B3 充电器 | ~$8 / ¥25-40 | Skip if you already have one |
| 12 | **Power switch + voltage display** | 电压显示开关 / 电池电量显示器 | ~$2 / ¥5-10 | Know when to charge |

---

## To Buy — Wiring & Misc

| # | Item (EN) | 淘寶搜尋關鍵字 | Est. Price | Notes |
|---|-----------|---------------|-----------|-------|
| 13 | **Dupont wires** (M-F, M-M, F-F) | 杜邦线 公对母 母对母 公对公 | ~$3 / ¥5-10 | Arduino ↔ motor driver ↔ servo connections |
| 14 | **USB-C cable** (short, 20-30cm) | USB-C 短线 数据线 20cm | ~$2 / ¥5 | Pi 5 power from buck converter |
| 15 | **USB3 cable** (short, for OAK-D) | USB3.0 短线 Type-C 30cm | ~$3 / ¥8 | OAK-D Lite connects to Pi 5 via USB3 |
| 16 | **Standoffs / screws kit** (M3) | M3 铜柱螺丝套装 / 尼龙柱螺丝 | ~$3 / ¥8-15 | Mount Pi 5, Arduino, camera to chassis |
| 17 | **Breadboard** (half-size) | 面包板 / 小面包板 | ~$1 / ¥3 | Prototyping. Replace with PCB later if needed |

---

## Approximate Cost Summary

| Category | Estimate |
|----------|----------|
| Sensors & Camera | ~$107 / ¥720 |
| Motors & Drive | ~$26-40 / ¥100-160 |
| Power | ~$28 / ¥100-140 |
| Wiring & Misc | ~$12 / ¥30-50 |
| **Total (excl. chassis/printer)** | **~$170-190 / ¥950-1070** |

---

## Forklift Structure — Approximate Design

```
                    SIDE VIEW
                    ─────────

        ┌─────────────────┐  ← Camera platform (OAK-D Lite + IR LEDs)
        │  OAK-D + IR     │
        └────────┬────────┘
                 │  ← Fork rails (vertical, 3D printed)
                 │     Servo + rack/gear lifts platform
                 │     Travel: ~15-30cm (sofa/bed height)
                 │
    ┌────────────┴────────────┐
    │                         │
    │   Upper deck             │  ← Pi 5 + Arduino + battery
    │   ┌─────┐ ┌─────┐      │
    │   │ Pi5 │ │ Batt│      │
    │   └─────┘ └─────┘      │
    │                         │
    ├─────────────────────────┤
    │   Lower chassis          │  ← Motor driver, buck converter
    │                         │
    ◯─────────────────────────◯  ← Drive wheels (with encoder motors)
              ◯                  ← Caster wheel (rear)


                    TOP VIEW
                    ────────

              ┌──────────┐
              │ Camera   │ ← Fork platform (can go up/down)
              │ OAK-D    │
              └────┬─────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    │  ┌─────┐         ┌──────┐  │
    │  │ Pi5 │         │ Batt │  │
    │  └─────┘         └──────┘  │
    │                             │
    │  ┌─────────┐   ┌────────┐  │
    │  │ Arduino │   │ Motor  │  │
    │  │         │   │ Driver │  │
    │  └─────────┘   └────────┘  │
    │                             │
    ◯───────────────────────────◯ ← L/R drive wheels
    │             ◯               │ ← Caster
    └─────────────────────────────┘

    Approx size: ~25cm long x 18cm wide x 15-35cm tall (fork range)


                FORK MECHANISM
                ──────────────

    ┌─────────┐
    │ Camera  │ ← Mounted on fork platform
    └────┬────┘
         │
    ═════╪═════  ← Platform slides on 2 vertical rails
    ║    │    ║
    ║  ┌─┴─┐  ║  ← Rack gear (齿条) attached to platform
    ║  │   │  ║
    ║  │ ⚙ │  ║  ← Pinion gear on servo/motor (齿轮)
    ║  │   │  ║     Servo rotates → rack moves up/down
    ║  └───┘  ║
    ║         ║  ← Vertical rails (光轴 / 线性导轨)
    ║         ║
    ╚═════════╝  ← Fixed to chassis base
```

### Fork Mechanism Options

| Option | 淘寶搜尋 | Pros | Cons |
|--------|---------|------|------|
| **Rack & pinion** (推薦) | 齿条齿轮模组 / 齿条升降机构 | Smooth, precise, easy to 3D print | Need to match gear sizes |
| **Lead screw** | 丝杆升降 T8 / 微型丝杆电机 | Very stable, self-locking (won't drop) | Slower, more expensive |
| **Pulley + belt** | 同步带 GT2 滑台 | Light, fast | Can slip under load |
| **Scissor lift** | 剪叉式升降台 微型 | Compact when lowered | Complex to 3D print, wobbly |

---

## Communication Architecture

```
┌──────────────────────────────────────────────────┐
│                   Pi 5 (Brain)                    │
│                                                  │
│  ┌──────────┐  ┌────────┐  ┌─────────────────┐  │
│  │ OAK-D    │  │ SLAM   │  │ Telegram Bot    │  │
│  │ Camera + │→ │ RTAB-  │  │ Commands:       │  │
│  │ YOLO     │  │ Map    │  │ "find Dan"      │  │
│  └──────────┘  └────────┘  │ "check water"   │  │
│                            │ "go to bedroom"  │  │
│  ┌──────────────────────┐  └────────┬────────┘  │
│  │ Navigation planner   │←──────────┘            │
│  │ A* path on SLAM map  │                        │
│  └──────────┬───────────┘                        │
│             │ Serial (USB)                       │
└─────────────┼────────────────────────────────────┘
              ↓
┌─────────────────────────────┐
│      Arduino (Body)          │
│  ┌─────────┐  ┌──────────┐  │
│  │ L/R     │  │ Fork     │  │
│  │ Motor   │  │ Servo    │  │
│  │ Control │  │ Control  │  │
│  └─────────┘  └──────────┘  │
└─────────────────────────────┘
```

---

## Taobao Search Tips

搜尋組合建議：
- 整車底盤: `智能小车底盘 编码器电机` or `ROS小车底盘`（如果想買現成底盤）
- 叉車玩具改裝: `遥控叉车 工程车 大号`（買來拆掉遙控器，換Arduino控制）
- OAK-D: 搜 `OAK-D Lite` 或到 Luxonis 官方淘寶店
- 3D列印材料: `PLA+ 耐摔` for chassis parts
