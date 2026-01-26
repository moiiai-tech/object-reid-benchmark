# Copyright (c) 2026 MoiiAi Inc. All rights reserved.
"""Benchmark runner for evaluating models with Rich progress bars."""

from __future__ import annotations

import time
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torchreid
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from ..datasets.registry import DatasetConfig

from ..models.base import BaseModelWrapper
from .logger import console


def evaluate_rank_batched(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
    batch_size: int = 500,
) -> tuple:
    """
    Memory-efficient evaluation of CMC and mAP in batches.

    Args:
        distmat: Distance matrix (N_q x N_g)
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        max_rank: Maximum rank to compute
        batch_size: Number of queries to process at once

    Returns:
        Tuple of (cmc, mAP)
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        console.print(
            f"[dim]Note: number of gallery samples is quite small, got {num_g}[/dim]"
        )

    console.print(
        f"[cyan]Evaluating CMC and mAP in batches (batch_size={batch_size})...[/cyan]"
    )

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    num_batches = (num_q + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating queries...", total=num_batches)

        for batch_idx in range(0, num_q, batch_size):
            end_idx = min(batch_idx + batch_size, num_q)
            batch_distmat = distmat[batch_idx:end_idx]
            batch_q_pids = q_pids[batch_idx:end_idx]
            batch_q_camids = q_camids[batch_idx:end_idx]

            indices = np.argsort(batch_distmat, axis=1)
            matches = (g_pids[indices] == batch_q_pids[:, np.newaxis]).astype(np.int32)

            for local_q_idx in range(len(batch_q_pids)):
                q_pid = batch_q_pids[local_q_idx]
                q_camid = batch_q_camids[local_q_idx]

                order = indices[local_q_idx]
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                keep = np.invert(remove)

                raw_cmc = matches[local_q_idx][keep]
                if not np.any(raw_cmc):
                    continue

                cmc = raw_cmc.cumsum()
                cmc[cmc > 1] = 1
                all_cmc.append(cmc[:max_rank])
                num_valid_q += 1

                num_rel = raw_cmc.sum()
                tmp_cmc = raw_cmc.cumsum()
                tmp_cmc = np.array([x / (i + 1.0) for i, x in enumerate(tmp_cmc)])
                tmp_cmc = tmp_cmc * raw_cmc
                AP = tmp_cmc.sum() / num_rel
                all_AP.append(AP)

            progress.update(task, advance=1)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def compute_distance_matrix_batched(
    qf: torch.Tensor,
    gf: torch.Tensor,
    batch_size: int = 2000,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute distance matrix in batches to avoid memory issues.

    Args:
        qf: Query features (N_q x D)
        gf: Gallery features (N_g x D)
        batch_size: Number of query samples to process at once
        metric: Distance metric (euclidean or cosine)

    Returns:
        Distance matrix (N_q x N_g) as numpy array
    """
    import tempfile
    import os

    num_q = qf.size(0)
    num_g = gf.size(0)

    # Calculate required memory
    required_gb = (num_q * num_g * 4) / (1024**3)  # 4 bytes per float32

    console.print(
        f"[cyan]Computing distance matrix in batches (batch_size={batch_size})...[/cyan]"
    )
    console.print(
        f"[dim]Query: {num_q}, Gallery: {num_g}, Total pairs: {num_q * num_g:,}[/dim]"
    )

    # If extremely large (> 100GB), use memory-mapped file
    if required_gb > 100:
        console.print(
            f"[yellow]⚠ Distance matrix would require {required_gb:.1f} GB RAM[/yellow]"
        )
        console.print(
            f"[cyan]💾 Using memory-mapped temporary file to save RAM...[/cyan]"
        )

        # Create temporary memory-mapped file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
        temp_filename = temp_file.name
        temp_file.close()

        distmat = np.memmap(
            temp_filename,
            dtype='float32',
            mode='w+',
            shape=(num_q, num_g)
        )
        console.print(f"[dim]Created memory-mapped file: {temp_filename}[/dim]")
    else:
        distmat = np.zeros((num_q, num_g), dtype=np.float32)

    num_batches = (num_q + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing distances...", total=num_batches)

        for i in range(0, num_q, batch_size):
            end_idx = min(i + batch_size, num_q)
            batch_qf = qf[i:end_idx]

            if metric == "euclidean":
                batch_dist = 2 - 2 * torch.mm(batch_qf, gf.t())
            elif metric == "cosine":
                batch_dist = 1 - torch.mm(batch_qf, gf.t())
            else:
                raise ValueError(f"Unknown metric: {metric}")

            distmat[i:end_idx, :] = batch_dist.cpu().numpy()
            progress.update(task, advance=1)

    # Note: If using memmap, the distmat is still memory-mapped
    # It will be cleaned up when the array is garbage collected
    # For extremely large datasets, this saves RAM at the cost of disk I/O
    if required_gb > 100:
        console.print(
            f"[dim]✓ Distance matrix computation complete (memory-mapped)[/dim]"
        )

    return distmat


def benchmark_model(
    model_wrapper: BaseModelWrapper,
    test_loader: Dict,
    dataset_config: DatasetConfig,
    gpu_id: int = 0,
) -> Dict[str, Any]:
    """
    Benchmark a single model on a dataset.

    Args:
        model_wrapper: Model wrapper instance
        test_loader: Dictionary with 'query' and 'gallery' DataLoaders
        dataset_config: Dataset configuration
        gpu_id: GPU ID to use

    Returns:
        Dictionary with benchmark results including mAP, CMC, time, etc.
    """
    from rich.panel import Panel
    from rich.text import Text

    header = Text()
    header.append("Benchmarking: ", style="bold cyan")
    header.append(model_wrapper.model_name, style="bold yellow")
    header.append(" on ", style="cyan")
    header.append(dataset_config.name, style="bold yellow")
    console.print(Panel(header, border_style="cyan", padding=(0, 2)))

    device = torch.device(f"cuda:{gpu_id}")

    num_params = model_wrapper.get_num_params()
    console.print(f"[green]Parameters:[/green] [bold]{num_params:,}[/bold]")

    console.print("[cyan]Extracting features from test data...[/cyan]")
    start_time = time.time()

    qf, q_pids, q_camids = [], [], []
    gf, g_pids, g_camids = [], [], []

    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            query_task = progress.add_task(
                "[cyan]Processing query images...", total=len(test_loader["query"])
            )
            for batch_idx, data in enumerate(test_loader["query"]):
                if isinstance(data, dict):
                    imgs, pids, camids = data["img"], data["pid"], data["camid"]
                else:
                    imgs, pids, camids = data[0], data[1], data[2]
                imgs = imgs.to(device)

                # Pass camera IDs to models that support SIE (CLIP-ReID, TransReID)
                # Models that don't use camera IDs will simply ignore this parameter
                features = model_wrapper.extract_features(imgs, cam_labels=camids)
                qf.append(features.cpu())
                q_pids.extend(pids.tolist())
                q_camids.extend(camids.tolist())
                progress.update(query_task, advance=1)

            gallery_task = progress.add_task(
                "[cyan]Processing gallery images...", total=len(test_loader["gallery"])
            )
            for batch_idx, data in enumerate(test_loader["gallery"]):
                if isinstance(data, dict):
                    imgs, pids, camids = data["img"], data["pid"], data["camid"]
                else:
                    imgs, pids, camids = data[0], data[1], data[2]
                imgs = imgs.to(device)

                # Pass camera IDs to models that support SIE (CLIP-ReID, TransReID)
                # Models that don't use camera IDs will simply ignore this parameter
                features = model_wrapper.extract_features(imgs, cam_labels=camids)
                gf.append(features.cpu())
                g_pids.extend(pids.tolist())
                g_camids.extend(camids.tolist())
                progress.update(gallery_task, advance=1)

    qf = torch.cat(qf, 0)
    gf = torch.cat(gf, 0)

    print(f"Query features shape: {qf.shape}")
    print(f"Gallery features shape: {gf.shape}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # NOTE: Features are already L2-normalized by all model wrappers (CLIP-ReID, TransReID,
    # DINOv2, DINOv3, SigLIP2, CLIP). No need to normalize again here.
    # All wrappers call F.normalize(features, p=2, dim=1) before returning.

    num_q = qf.size(0)
    num_g = gf.size(0)
    total_pairs = num_q * num_g

    use_batched = total_pairs > 50_000_000 or num_g > 50_000

    if use_batched:
        console.print(
            f"[yellow]Large dataset detected ({num_q} x {num_g} = {total_pairs:,} pairs)[/yellow]"
        )
        console.print(
            "[cyan]Using memory-efficient batched distance computation...[/cyan]"
        )

        # Adjust batch size based on gallery size to avoid GPU OOM
        if num_g > 1_000_000:
            # For extremely large galleries (> 1M), use very small batches
            batch_size = 100
            console.print(f"[dim]Using reduced batch size ({batch_size}) for huge gallery[/dim]")
        elif num_g > 100_000:
            batch_size = 500
        else:
            batch_size = 2000

        distmat = compute_distance_matrix_batched(
            qf, gf, batch_size=batch_size, metric="euclidean"
        )
    else:
        console.print("[cyan]Computing distance matrix...[/cyan]")
        distmat = torchreid.metrics.compute_distance_matrix(qf, gf, metric="euclidean")
        distmat = distmat.cpu().numpy()

    q_pids = np.array(q_pids)
    q_camids = np.array(q_camids)
    g_pids = np.array(g_pids)
    g_camids = np.array(g_camids)

    if use_batched or num_q > 5000:
        cmc, mAP = evaluate_rank_batched(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=50,
            batch_size=500,
        )
    else:
        console.print("[cyan]Computing CMC and mAP...[/cyan]")
        cmc, mAP = torchreid.metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=50,
            use_metric_cuhk03=dataset_config.use_cuhk03_metric,
        )

    elapsed_time = time.time() - start_time

    # Display results in a nice format
    from rich.table import Table

    results_table = Table(
        title="Evaluation Results", show_header=True, header_style="bold green"
    )
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", justify="right", style="yellow")

    results_table.add_row("mAP", f"{mAP * 100:.1f}%")
    results_table.add_row("Rank-1", f"{cmc[0] * 100:.1f}%")
    results_table.add_row("Rank-5", f"{cmc[4] * 100:.1f}%")
    results_table.add_row("Rank-10", f"{cmc[9] * 100:.1f}%")
    results_table.add_row("Rank-20", f"{cmc[19] * 100:.1f}%")
    results_table.add_row("Time", f"{elapsed_time:.2f}s")

    console.print(results_table)
    console.print(
        f"[green]Evaluation completed in {elapsed_time:.2f} seconds[/green]\n"
    )

    return {
        "model": model_wrapper.model_name,
        "dataset": dataset_config.name,
        "params": num_params,
        "time": elapsed_time,
        "mAP": mAP * 100,
        "Rank-1": cmc[0] * 100,
        "Rank-5": cmc[4] * 100,
        "Rank-10": cmc[9] * 100,
        "Rank-20": cmc[19] * 100,
    }
