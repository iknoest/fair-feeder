# Fair Feeder — Presentation Slides

Copy each slide's content into your presentation tool (Google Slides, Keynote, PowerPoint).
Each `---` is a slide break.

---

## Slide 1: Title

**Fair Feeder**
*How I Built an AI Cat Feeding Monitor with a $15 Camera*

<!-- PHOTO: Dan and Sanbo together -->

---

## Slide 2: The Problem

**"Did Dan eat enough today?"**

- Two cats: Dan (picky eater) and Sanbo (food thief)
- Morning feeding at 6am — I'm half-asleep
- Can't watch the bowl 24/7
- Need: a system that watches for me and reports back

<!-- PHOTO: Dan and Sanbo side by side -->

---

## Slide 3: The Setup ($80 total)

| | |
|---|---|
| Tapo C210 | $15 — IR camera, 2K, overhead mount |
| Raspberry Pi 5 | $60 — 24/7 recording + upload |
| Google Drive | Free — video storage |
| Colab / GitHub Actions | Free — AI analysis |
| Telegram Bot | Free — daily reports |

<!-- PHOTO: Camera mounted above bowl, Pi nearby -->

---

## Slide 4: How It Connects

```
Camera (RTSP) → Pi (records motion) → Google Drive → Cloud AI → Telegram → My Phone
```

<!-- Use the hardware Mermaid diagram from the blog post -->

---

## Slide 5: What the AI Sees

5 detection classes:
- **Dan** (tuxedo cat)
- **Sanbo** (calico cat)
- **Dan_hand** (my hand feeding)
- **Bowl** (food bowl)
- **Kibble** (each piece of food)

<!-- PHOTO: Annotated video frame with bounding boxes -->

---

## Slide 6: The Pipeline

```
Video → YOLOv11 (3 min) → Cache → Analytics (2 sec) → Telegram Report
```

Key design: AI runs once, analytics are instant and re-runnable.

---

## Slide 7: What Failed

| Attempt | Result |
|---------|--------|
| Pre-trained models on Pi | Saw "ovens" and "sinks" instead of cats |
| Full YOLOv11 on Pi | Too slow (seconds per frame) |
| Counting kibble naively | Same kibble counted 2-3x due to occlusion |

**Lesson: General models don't work for unusual angles. Custom training is necessary.**

---

## Slide 8: What Worked

| Solution | Result |
|----------|--------|
| Custom YOLOv11 on free GPU | 95%+ accuracy |
| Split: Pi records, Cloud analyzes | Best of both devices |
| Peak kibble count for estimation | Correct count despite occlusion |
| Rolling median filter | Smooth out flickering |

---

## Slide 9: The Hardest Problem — Counting Kibble

```
Cat arrives → 26 kibble visible
Cat eating (blocks view) → 8 visible (hidden, not eaten!)
Cat shifts → 20 visible (same kibble, now visible again)
Cat leaves → 2 visible

Naive: 18 + 18 = 36 eaten
Reality: 26 - 2 = 24 eaten
```

**Solution: Use the peak visible count as the true starting amount.**

---

## Slide 10: The Data Flywheel (Self-Improving AI)

```
Daily report → Auto-flag mistakes → Upload to Roboflow → Human corrects (30 min) → Retrain → Deploy
```

The system finds its own mistakes:
- Single-frame hallucinations
- Contradicting detections
- Impossible scenarios (hand without cat)

<!-- Use the flywheel Mermaid diagram -->

---

## Slide 11: Results — V13 vs V14

| | V13 | V14 |
|---|---|---|
| Sanbo hallucinations | 18/19 videos | **0** |
| False hand-feeding | 8-20/video | **0** |
| Overall accuracy | 95.6% | **95.7%** |

One flywheel cycle. 231 frames reviewed. 30 minutes of human work.

---

## Slide 12: My Daily Experience

```
7:00 AM — Phone buzzes

"Dan ate well — no compensation needed"
Dan: 24 kibble (100%)
Sanbo: 0 kibble (0%)

→ I go about my day, knowing Dan is fed.
```

<!-- PHOTO: Telegram report screenshot -->

---

## Slide 13: Architecture

<!-- Use the full architecture Mermaid diagram -->

4 layers: Hardware → Storage → AI → Output
+ Self-improving feedback loop

---

## Slide 14: What's Next — Autonomous Monitor

**Current:** Fixed camera, fixed location
**Future:** Mobile unit, self-charging, multi-room

- Mobile platform follows the cats
- Onboard AI (no cloud dependency)
- Self-charging dock
- Multi-cat, multi-bowl monitoring

All existing code (model, tracker, flywheel) transfers directly.

<!-- PHOTO: Concept sketch of mobile unit -->

---

## Slide 15: Key Takeaways

1. **Start cheap** — $15 camera + free GPU = 95%+ accuracy
2. **Split by strengths** — Pi records 24/7, Cloud analyzes on-demand
3. **Build a flywheel** — The model improves itself every week
4. **Solve your own problems** — The best projects come from real frustration
5. **Iterate fast** — 37 bugs fixed, 50+ decisions, multiple model versions

---

## Slide 16: Closing

*"I didn't set out to learn YOLO or Roboflow — I just wanted to know if Dan ate enough."*

<!-- PHOTO: Dan and Sanbo relaxing — the happy ending -->

**GitHub:** [github.com/iknoest/fair-feeder](https://github.com/iknoest/fair-feeder)

---

# Speaker Notes

**Slide 2:** Start with the story — make it relatable. Everyone with pets understands the worry.

**Slide 7:** This is the "honesty" slide — show that the path wasn't straight. Pre-trained models failing at unusual angles is a great surprise for the audience.

**Slide 9:** The occlusion problem is the most interesting technical challenge. Use the animation: cat arrives (26) → blocks view (8) → moves (20) → leaves (2). Ask the audience: "How many did the cat eat?" before revealing the answer.

**Slide 10:** The flywheel is the "wow" moment. Emphasize: the system finds its own mistakes, uploads them, and you just correct labels for 30 minutes. Then retrain. V14 was dramatically better after just one cycle.

**Slide 14:** End with vision — the movable monitor shows this isn't just a hobby project, it's a platform with real scaling potential.
