"""Shared inputs and network architecture for the walking and straight-line forecast comparison."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
hours = pd.date_range("2023-01-01", "2025-01-01", freq="h", inclusive="left")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS, SEED = 8192, 4, 42


def count_departures(group):
    ids = pd.Index(sorted(group.unique()), name="군집번호")
    verified = pd.read_csv(ROOT / "전처리/월별 집계 검증.csv").set_index("월")
    sample_ids = pd.read_csv(ROOT / "전처리/선정 대여소.csv").station_id
    totals = np.zeros((len(hours), len(ids)), dtype=np.int32)
    # 기준선에서 시차 대여량 중 예측 정시에 아직 반납이 확인되지 않은 건수를 뺀다.
    late_returns = {lag: np.zeros_like(totals) for lag in (24, 168)}
    # 각 출발 시간에 대해 1~24시간 후까지 확인된 누적 대여 건수를 저장한다.
    confirmed = np.empty((len(hours), len(ids), 24), dtype=np.int16)
    with TemporaryDirectory(prefix="cluster_demand_") as tmp:
        cached = os.environ.get("BIKE_AGGREGATE_CACHE")
        cache = Path(cached) if cached else Path(tmp)
        if not cached:
            subprocess.run([sys.executable, str(ROOT / "전처리/rental_aggregate.py"), str(cache)], check=True)
        for month, ref in verified.iterrows():
            start = hours.get_loc(pd.Timestamp(month))
            with np.load(cache / f"{month}.npz") as z:
                station_ids, counts = z["ids"], z["counts"]
            nh = counts.shape[1]
            assert nh == pd.Timestamp(month).days_in_month * 24
            assert counts.sum() == ref["집계 대여 건수"]
            assert counts[np.isin(station_ids, sample_ids)].sum() == ref["표본 50곳 대여 건수"]
            groups = group.reindex(station_ids)
            matched = groups.notna().to_numpy()
            cluster_idx = ids.get_indexer(groups[matched])
            np.add.at(totals[start:start+nh].T, cluster_idx, counts[matched])
            assigned = int(totals[start:start+nh].sum())
            assert assigned == counts[matched].sum()
            events = np.memmap(cache / f"{month}.bin", dtype="<i4", mode="r").reshape(-1, 3)
            assert len(events) == ref["필터 후 원천 행"] and np.count_nonzero(events[:,0]) == counts.sum()
            for lag, correction in late_returns.items():
                mask = (events[:,0] > 0) & (events[:,2] > events[:,1] + lag)
                delayed = events[mask, :2]
                positions = ids.get_indexer(group.reindex(delayed[:,0]))
                keep = (positions >= 0) & (delayed[:,1] + lag < len(hours))
                np.add.at(correction, (delayed[keep,1] + lag, positions[keep]), 1)
            # 반납 확인이 출발 다음 정시보다 늦은 대여만 지연 시간별로 센다.
            bins = np.zeros((26, nh, len(ids)), dtype=np.int32)
            for block_start in range(0, len(events), 250_000):
                block = events[block_start:block_start+250_000]
                delay = block[:,2] - block[:,1]
                delayed = (block[:,0] > 0) & (delay > 1)
                group_pos = ids.get_indexer(group.reindex(block[delayed,0]))
                keep = group_pos >= 0
                np.add.at(bins, (np.minimum(delay[delayed][keep],25),
                                 block[delayed,1][keep] - start, group_pos[keep]), 1)
            pending = np.zeros((nh, len(ids)), dtype=np.int32)
            monthly_counts = totals[start:start+nh]
            assert monthly_counts.max() <= np.iinfo(np.int16).max
            for lag in range(24,0,-1):
                pending += bins[lag+1]
                as_known = monthly_counts - pending
                assert as_known.min() >= 0
                confirmed[start:start+nh,:,lag-1] = as_known
            del bins, block, events

    return totals, late_returns, confirmed

def count_arrivals(group):
    from cluster_returns import build_cluster_returns
    cache_path = ROOT / "전처리/전체 대여소 시간별 반납.npz"
    if not cache_path.is_file():
        build_cluster_returns(group.index.to_series(), cache_path)
    ids = pd.Index(sorted(group.unique()))
    with np.load(cache_path) as data:
        np.testing.assert_array_equal(data["ids"], group.index)
        np.testing.assert_array_equal(data["station_ids"], group.index)
        np.testing.assert_array_equal(data["station_groups"], group.index)
        counts = data["counts"]
    result = np.zeros((len(hours), len(ids)), dtype=np.int32)
    np.add.at(result.T, ids.get_indexer(group), counts.T)
    np.testing.assert_array_equal(result.sum(axis=1),counts.sum(axis=1))
    assert result.sum() == 84684782
    return result

VARIANTS = {
    "base": ("없음", []),
    "both": ("둘 다", ["returns_prev_1h","returns_prev_3h","weather_match_mean","weather_match_missing"]),
    "returns_only": ("반납량만", ["returns_prev_1h", "returns_prev_3h"]),
    "weather_match_only": ("유사 기상만", ["weather_match_mean", "weather_match_missing"]),
}

class ClusterNet(nn.Module):
    def __init__(self, kind, context_size, extra_size, n_clusters):
        super().__init__()
        self.embedding = nn.Embedding(n_clusters, 8)
        self.gru = nn.GRU(1,16,batch_first=True) if kind == "GRU" else None
        history_size = 16 if self.gru is not None else 24
        self.head = nn.Sequential(nn.Linear(history_size + context_size + 8 + 1 + extra_size, 64), nn.ReLU(),
                                  nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Softplus())

    def forward(self, seq, historical_features, context, g):
        history = seq
        if self.gru is not None:
            _, hidden = self.gru(seq.unsqueeze(-1))
            history = hidden[-1]
        features = torch.cat([history,historical_features,context,self.embedding(g)],dim=1)
        return self.head(features).squeeze(1)


def prepare_inputs(group, totals, late_returns, confirmed):
    """Return the same as-of inputs for any partition of the 2,766 stations."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ids = pd.Index(sorted(group.unique()))
    N = len(ids)
    deltas = torch.arange(24,0,-1,device=DEVICE)
    np.testing.assert_array_equal(confirmed[:-24,:,23], totals[:-24] - late_returns[24][24:])
    weekly = np.zeros_like(totals)
    weekly[168:] = totals[:-168] - late_returns[168][168:]
    assert weekly.min() >= 0 and weekly.max() <= np.iinfo(np.int16).max
    past = torch.as_tensor(confirmed, device=DEVICE)
    week = torch.as_tensor(weekly.astype(np.int16), device=DEVICE)
    actual = torch.as_tensor(totals, device=DEVICE)
    weather_names = ["temperature","humidity","precipitation","windspeed"]
    wx = pd.read_csv(ROOT / "분석용 데이터/Seoul_public_bicycle.csv",parse_dates=["datetime"]).set_index("datetime")
    w = wx[weather_names].reindex(hours - pd.Timedelta(hours=1)).to_numpy(dtype="float32")
    fixed = np.column_stack([np.eye(24)[hours.hour], np.eye(7)[hours.weekday],
                             np.eye(12)[hours.month-1], wx.holiday.loc[hours].to_numpy()]).astype("float32")
    def context_at(cutoff):
        reference = w[168:cutoff-23]
        median = np.nanmedian(reference,axis=0)
        reference = np.where(np.isnan(reference),median,reference)
        mean, std = reference.mean(axis=0), reference.std(axis=0).clip(1e-6)
        values = (np.where(np.isnan(w),median,w)-mean)/std
        combined = np.column_stack([fixed,values,np.isnan(w)])
        assert np.isfinite(combined).all()
        return torch.as_tensor(combined,device=DEVICE), {"median":median.tolist(),"mean":mean.tolist(),"std":std.tolist()}
    returned = count_arrivals(group)
    assert returned.shape == totals.shape and returned.min() >= 0 and returned.max() <= 32767
    arrivals = torch.as_tensor(returned.astype(np.int16),device=DEVICE)

    limits = np.array([5.,15.,2.,2.])  # temperature, humidity, precipitation, windspeed
    assert weather_names == ["temperature","humidity","precipitation","windspeed"]
    complete = np.isfinite(w).all(axis=1)
    wet = w[:,2] > 0
    off_day = (hours.weekday >= 5) | wx.holiday.loc[hours].eq(1).to_numpy()
    weather_avg = np.full(totals.shape, np.nan, dtype=np.float32)
    match_count = np.zeros(len(hours),dtype="int8")
    offsets = np.arange(1,366)*24
    for t in range(len(hours)):
        if not complete[t]:
            continue
        prev = t-offsets[offsets<=t]
        prev = prev[complete[prev] & (off_day[prev]==off_day[t]) & (wet[prev]==wet[t])]
        differences = np.abs(w[prev]-w[t])/limits
        close = (differences <= 1).all(axis=1)
        distances = (differences[close]**2).sum(axis=1)
        selected = prev[close][np.argsort(distances,kind="stable")[:5]]
        if len(selected):
            assert (selected+24 <= t).all()
            match_count[t] = len(selected)
            weather_avg[t] = confirmed[selected,:,23].mean(axis=0)
    assert np.array_equal(np.isnan(weather_avg).all(axis=1),match_count==0)
    matched_avg = torch.as_tensor(weather_avg,device=DEVICE)
    missing_match = torch.as_tensor(match_count==0,device=DEVICE)

    def batch_inputs(flat, variant):
        t = flat // N
        g = flat % N
        seq = past[t[:,None]-deltas,g[:,None],deltas-1].float().log1p()
        weekly = week[t,g].float().log1p()
        values = [weekly]
        if variant in ("returns_only","both"):
            last = arrivals[t-1,g].float()
            last_three = last + arrivals[t-2,g] + arrivals[t-3,g]
            values.extend([last.log1p(),last_three.log1p()])
        if variant in ("weather_match_only","both"):
            values.extend([matched_avg[t,g].nan_to_num().log1p(),missing_match[t].float()])
        historical = torch.stack(values,dim=1)
        return t, g, seq, historical


    return context_at, batch_inputs, past, actual, weather_names
