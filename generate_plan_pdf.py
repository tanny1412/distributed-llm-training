from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER

doc = SimpleDocTemplate(
    "training_plan.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("title", fontSize=18, fontName="Helvetica-Bold", spaceAfter=6)
subtitle_style = ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica", textColor=colors.grey, spaceAfter=16)
h2_style = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica", spaceAfter=4, leading=15)
code_style = ParagraphStyle("code", fontSize=9, fontName="Courier", spaceAfter=4, leading=13, textColor=colors.HexColor("#2d2d2d"), backColor=colors.HexColor("#f4f4f4"), leftIndent=10, rightIndent=10)
label_style = ParagraphStyle("label", fontSize=10, fontName="Helvetica-Bold", spaceAfter=2)
badge_style = ParagraphStyle("badge", fontSize=9, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER)
stage_title_style = ParagraphStyle("stage_title", fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a1a"))
stage_sub_style = ParagraphStyle("stage_sub", fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#555555"))

STAGE_COLORS = [
    colors.HexColor("#2563EB"),  # blue   — Stage 1
    colors.HexColor("#16A34A"),  # green  — Stage 2
    colors.HexColor("#9333EA"),  # purple — Stage 3
    colors.HexColor("#EA580C"),  # orange — Stage 4
]

def stage_header(n, title, subtitle):
    badge = Table(
        [[Paragraph(f"Stage {n}", badge_style)]],
        colWidths=[1.8*cm],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), STAGE_COLORS[n - 1]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    text = Table(
        [[Paragraph(title, stage_title_style)],
         [Paragraph(subtitle, stage_sub_style)]],
        colWidths=[14.2*cm],
    )
    text.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    row = Table([[badge, text]], colWidths=[1.8*cm, 14.2*cm])
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 1, STAGE_COLORS[n - 1]),
    ]))
    return row

def stage_table(rows):
    t = Table(rows, colWidths=[5.5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t

story = []

story.append(Paragraph("Distributed LLM Training — Run Plan", title_style))
story.append(Paragraph("Llama-3-8B · 4× A100 SXM 80GB · RunPod · MLflow tracking", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd"), spaceAfter=12))

# Stage 1
story.append(Spacer(1, 8))
story.append(stage_header(1, "Single GPU — Peak Memory Experiment", "Goal: measure actual HBM needed at any moment to decide which GPU tier fits each configuration."))
story.append(Spacer(1, 8))
story.append(Spacer(1, 6))
story.append(stage_table([
    ["Run 1", "Gradient checkpointing ON\npython train_single.py"],
    ["Run 2", "Gradient checkpointing OFF\n(comment out gradient_checkpointing_enable(), change run_name)"],
    ["Metrics logged", "samples_per_sec · steady_memory_mb · peak_memory_mb"],
    ["Key formula", "peak(OFF) − peak(ON) = activation memory saved by checkpointing\nsavings % = (peak(OFF) − peak(ON)) / peak(OFF) × 100%"],
    ["Decision", "peak_memory_mb = minimum GPU HBM needed\nif peak(ON) fits a cheaper GPU → checkpointing pays off\ncost = 18% throughput penalty"],
]))
story.append(Spacer(1, 4))
story.append(Paragraph("Why peak and not steady-state:", label_style))
story.append(Paragraph("Steady-state (after optimizer.step) only shows weights + gradients + optimizer states — activations already freed. OOMs happen during the forward→backward window. Peak memory right after backward() is the true ceiling.", body_style))

# Stage 2
story.append(Spacer(1, 8))
story.append(stage_header(2, "DDP — Throughput Scaling", "Goal: measure throughput gains and communication overhead as GPUs increase. Single GPU throughput from Stage 1 is the baseline."))
story.append(Spacer(1, 8))
story.append(stage_table([
    ["Commands", "torchrun --nproc_per_node=2 train_ddp.py\ntorchrun --nproc_per_node=4 train_ddp.py\n(1 GPU skipped — identical to single GPU baseline)"],
    ["Batch size", "Fixed at 4 across all runs (clean comparison)"],
    ["Metrics logged", "samples_per_sec · peak_memory_mb per rank"],
    ["Key formula", "scaling efficiency = (N_gpu_throughput / (N × 1gpu_throughput)) × 100%\ntarget: 80–92% at 4 GPUs on NVLink"],
    ["Memory expectation", "Peak per rank = same as single GPU\nDDP does not shard — full model on every rank"],
    ["Throughput expectation", "~1.8× at 2 GPUs, ~3.4× at 4 GPUs\n(not linear — all-reduce communication overhead)"],
]))
story.append(Spacer(1, 4))
story.append(Paragraph("DDP answers: how much throughput do you gain by adding GPUs? Memory problem unchanged.", body_style))

# Stage 3
story.append(Spacer(1, 8))
story.append(stage_header(3, "FSDP — Memory Savings + Throughput Recovery", "Goal: show memory drop from sharding. Then use freed memory for larger batches to recover throughput."))
story.append(Spacer(1, 8))
story.append(stage_table([
    ["Commands", "torchrun --nproc_per_node=4 train_fsdp.py  (batch=4)\n"
                 "torchrun --nproc_per_node=4 train_fsdp.py  (batch=16)\n"
                 "torchrun --nproc_per_node=2 train_fsdp.py  (batch=16)"],
    ["Gradient checkpointing", "OFF — sharding alone should be enough on A100 80GB"],
    ["Metrics logged", "samples_per_sec · peak_memory_mb · steady_memory_mb"],
    ["Memory expectation", "Peak per rank dramatically lower than DDP\nFSDP shards weights + gradients + optimizer states across 4 GPUs"],
    ["Throughput story", "FSDP 4GPU batch=4   → lower than DDP (communication overhead)\n"
                         "FSDP 4GPU batch=16  → back to ~DDP level (bigger batch amortizes cost)\n"
                         "FSDP 2GPU batch=16  → can 2 GPUs match DDP 4 GPU throughput?"],
    ["Key insight", "DDP is memory-constrained on batch size.\n"
                    "FSDP frees memory → bigger batch → recovers throughput lost to communication.\n"
                    "FSDP 2 GPUs ≈ DDP 4 GPUs in throughput = same job, half the GPU cost."],
]))

# Stage 4
story.append(Spacer(1, 8))
story.append(stage_header(4, "Ray Train — Managed DDP", "Goal: same throughput as DDP with simpler setup. No manual process group init or MASTER_ADDR config."))
story.append(Spacer(1, 8))
story.append(stage_table([
    ["Command", "python train_ray.py"],
    ["Memory expectation", "Same as DDP — Ray Train wraps DDP, same memory math"],
    ["Throughput expectation", "Same as DDP — same underlying communication"],
    ["What changes", "No manual dist.init_process_group\nNo MASTER_ADDR/PORT setup\nNative HPO integration via Ray Tune"],
    ["Decision", "Same numbers, simpler setup. Worth it for large clusters\nwhere multi-node coordination is the hard part."],
]))

# Summary table
story.append(Spacer(1, 8))
story.append(Paragraph("Expected Results Summary", h2_style))
summary = [
    ["Stage", "GPUs", "Batch", "Peak Mem/rank", "Throughput", "Key finding"],
    ["Single GPU (ckpt ON)", "1", "4", "TBD", "4.61 s/s", "Baseline"],
    ["Single GPU (ckpt OFF)", "1", "4", "TBD", "5.45 s/s", "peak diff = activation cost"],
    ["DDP", "2", "4", "~same as single", "TBD", "scaling efficiency at 2 GPUs"],
    ["DDP", "4", "4", "~same as single", "TBD", "scaling efficiency at 4 GPUs"],
    ["FSDP", "4", "4", "TBD (much lower)", "TBD", "apples-to-apples vs DDP"],
    ["FSDP", "4", "16", "TBD (much lower)", "TBD", "recover throughput with freed memory"],
    ["FSDP", "2", "16", "TBD (much lower)", "TBD", "2 GPUs ≈ DDP 4 GPUs?"],
    ["Ray Train", "4", "4", "~same as DDP", "TBD", "setup simplicity vs torchrun"],
]
t = Table(summary, colWidths=[3.5*cm, 1.5*cm, 1.5*cm, 2.8*cm, 2.5*cm, 4.7*cm])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(t)

# compare_runs.py section
story.append(Spacer(1, 8))
story.append(Paragraph("Tracking Results — compare_runs.py", h2_style))
story.append(Paragraph(
    "After each stage, run compare_runs.py on the pod to auto-calculate scaling efficiency and peak memory savings. "
    "Reads all runs from MLflow — runs not yet completed show TBD.",
    body_style
))
story.append(Spacer(1, 6))
story.append(stage_table([
    ["Run on pod", "python compare_runs.py\n(MLflow server must be running at localhost:8080)"],
    ["Memory table", "peak_memory_mb · steady_memory_mb · activation_memory per run\n"
                     "checkpointing savings % = (peak_OFF − peak_ON) / peak_OFF × 100%"],
    ["Scaling table", "DDP and FSDP only\n"
                      "expected = single_gpu_throughput × world_size\n"
                      "actual multiplier = run_throughput / single_gpu_throughput\n"
                      "efficiency % = (actual / expected) × 100%"],
    ["Partial runs OK", "Script works at any stage — missing runs print TBD\n"
                        "Run after single GPU → after DDP → after FSDP to see table fill progressively"],
]))
story.append(Spacer(1, 6))
story.append(Paragraph("GPU sizing decision from peak memory:", label_style))
story.append(Paragraph(
    "peak(checkpointing OFF) = minimum GPU HBM without tricks. "
    "peak(checkpointing ON) = minimum GPU HBM with checkpointing. "
    "If the cheaper GPU tier fits peak(ON), checkpointing pays off — cost is 18% throughput.",
    body_style
))

doc.build(story)
print("training_plan.pdf generated.")
