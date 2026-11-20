"""실행: python preprocess.py. pandas, numpy, openpyxl 필요."""
import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from rental_aggregate import INPUT, ROOT, run_aggregation

HOURS = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", freq="h")
TRAIN_END = pd.Timestamp("2024-07-01")
TEST_START = pd.Timestamp("2024-10-01")
MONTHS = list(pd.period_range("2023-01", "2024-12", freq="M").astype(str))
WEATHER = ["temperature", "humidity", "precipitation", "windspeed", "dew_point", "sunshine",
           "solar_radiation", "snow_depth", "cloud_cover", "visibility", "ground_temp"]
LAGS = (*range(1, 25), 168)
WINDOWS = (3, 24, 168)
GROUP_SIZES = {"low": 17, "medium": 16, "high": 17}
OUT = Path(__file__).resolve().parent


def snapshot(month):
    path = ROOT / "따릉이 대여소 비교" / "서울시 원본" / f"공공자전거 대여소 정보 {month}.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    records = {}
    try:
        for row in workbook.active.iter_rows(min_row=6, max_col=10, values_only=True):
            if not str(row[0]).strip().isdigit():
                continue
            number = int(row[0])
            if number in records:
                raise ValueError(f"현황표 중복 번호 {number}")
            name, district, address = (" ".join(str(x).split()) for x in row[1:4])
            records[number] = (name, district, address, float(row[4]), float(row[5]),
                               str(row[6])[:10], row[7] or 0, row[8] or 0, str(row[9]).strip())
    finally:
        workbook.close()
    return records


def unchanged(a, b):
    if a[:3] != b[:3] or a[5:] != b[5:]:
        return False
    lat1, lon1, lat2, lon2 = map(math.radians, (a[3], a[4], b[3], b[4]))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12_742_000 * math.asin(math.sqrt(h)) < 1.0


def select_stations(cache):
    # 2024년 6월 현황표는 학습 종료 후인 7월에 공개되었다.
    maps = [snapshot(m) for m in ("2022-12", "2023-06", "2023-12")]
    ids = sorted(k for k in set.intersection(*(set(d) for d in maps))
                 if all(unchanged(a[k], b[k]) for a, b in zip(maps, maps[1:])))
    frames = []
    for month in MONTHS[:18]:
        with np.load(cache / (month + ".npz")) as z:
            daily = z["counts"].reshape(len(z["ids"]), -1, 24).sum(axis=2)
            frames.append(pd.DataFrame(daily, index=z["ids"]).reindex(ids, fill_value=0).to_numpy())
    days = np.concatenate(frames, axis=1)
    training = HOURS[HOURS < TRAIN_END]
    assert days.shape[1] * 24 == len(training)
    eligible = []
    for station, daily in zip(ids, days):
        active = np.flatnonzero(daily)
        longest = int(np.diff(np.r_[-1, active, len(daily)]).max()-1)
        if longest >= 7:  # 매월 대여가 있으며 연속 일주일 이상 기록이 끊기지 않음
            continue
        meta = maps[-1][station]
        eligible.append((station, meta[0], meta[1], meta[3], meta[4], meta[6]+meta[7],
                         int(daily.sum()), daily.sum()/len(training), longest))
    fields = ["station_id", "station_name", "district", "latitude", "longitude",
              "racks_2023_12", "train_departures", "train_hourly_mean", "train_max_zero_days"]
    candidates = pd.DataFrame(eligible, columns=fields).sort_values(["train_departures", "station_id"])
    selected = []
    for (label, size), indices in zip(GROUP_SIZES.items(), np.array_split(np.arange(len(candidates)), 3)):
        group = candidates.iloc[indices].sort_values("station_id").copy()
        if len(group) < size:
            raise ValueError("표본 부족")
        midpoints = ((np.arange(10)+0.5)*len(group)/10).astype(int)
        group["demand_group"] = label
        group["sample_group"] = "추가 20곳"
        group.loc[group.iloc[midpoints].index, "sample_group"] = "기존 30곳"
        remaining = np.setdiff1d(np.arange(len(group)), midpoints)
        extra = size - len(midpoints)
        positions = ((np.arange(extra) + 0.5) * len(remaining) / extra).astype(int)
        selected.append(group.iloc[np.r_[midpoints, remaining[positions]]])
    stations = pd.concat(selected).sort_values("station_id").reset_index(drop=True)
    assert len(stations)==stations.station_id.nunique()==sum(GROUP_SIZES.values())
    return stations, len(candidates)


def calendar_and_weather():
    source = pd.read_csv(INPUT / "Seoul_public_bicycle.csv", parse_dates=["datetime"]).set_index("datetime")
    if not source.index.is_unique or not HOURS.isin(source.index).all():
        raise ValueError("기상 시간축 오류")
    wx_hours = HOURS - pd.Timedelta(hours=1)
    if not wx_hours.isin(source.index).all():
        raise ValueError("기상 관측시각 누락")
    calendar = pd.DataFrame(index=HOURS)
    calendar["split"] = np.select([HOURS < TRAIN_END, HOURS < TEST_START],
                                  ["train", "validation"], default="test")
    calendar["hour"] = HOURS.hour
    calendar["weekday"] = HOURS.dayofweek
    calendar["month"] = HOURS.month
    calendar["is_weekend"] = (HOURS.dayofweek >= 5).astype(int)
    holiday = source.loc[HOURS,"holiday"]
    if not holiday.isin([0,1]).all():
        raise ValueError("공휴일 표시 오류")
    calendar["is_holiday"] = holiday.to_numpy()
    calendar["weather_time"] = wx_hours
    for name in WEATHER:
        values = source.loc[wx_hours, name].to_numpy()
        calendar[name+"_lag1h"] = values
        calendar[name+"_missing"] = pd.isna(values).astype(int)
    return calendar


def collect_events(cache, ids):
    events = {int(i): [] for i in ids}
    city = []
    selected_counts = []
    for month in MONTHS:
        with np.load(cache / (month + ".npz")) as z:
            mx = pd.DataFrame(z["counts"], index=z["ids"])
            city.extend(mx.sum(axis=0).to_numpy())
            selected_counts.append(mx.reindex(ids, fill_value=0).to_numpy())
        raw = np.memmap(cache / (month + ".bin"), dtype="<i4", mode="r").reshape(-1, 3)
        hits = raw[np.isin(raw[:,0], ids)]
        for i in ids:
            events[i].append(hits[hits[:,0] == i, 1:])
        del raw
    if min(city) == 0 or len(city) != len(HOURS):
        raise ValueError("서울 전체 시간대 대여 누락")
    return events, np.concatenate(selected_counts, axis=1)


def past_counts(dep, available):
    # 한 대여가 확정되는 최초 예측 정시 이후에만 과거 실적에 포함한다.
    n = len(HOURS)
    result = {}
    for lag in LAGS:
        t = dep + lag
        ok = (t < n) & (available <= t)
        values = np.bincount(t[ok], minlength=n).astype(float)
        values[:lag] = np.nan
        result[f"rentals_lag_{lag}h"] = values
    for win in WINDOWS:
        first = np.maximum(dep+1, available)
        last = np.minimum(dep+win, n-1)
        ok = first <= last
        diff = (np.bincount(first[ok], minlength=n+1) -
                np.bincount(last[ok]+1, minlength=n+1))
        rolling = diff.cumsum()[:n]
        assert (rolling >= 0).all()
        means = rolling / win
        means[:win] = np.nan
        result[f"rentals_mean_{win}h"] = means
    return result


def build(cache):
    stations, pool_size = select_stations(cache)
    calendar = calendar_and_weather()
    ids = stations.station_id.to_numpy()
    events, counts = collect_events(cache, ids)
    frames, gaps = [], []
    for i, station in enumerate(ids):
        e = np.concatenate(events[station])
        dep, available = e[:,0].astype("int64"), e[:,1].astype("int64")
        rentals = counts[i]
        if not np.array_equal(np.bincount(dep,minlength=len(HOURS)), rentals):
            raise ValueError("대여소 원천 합계 검증 실패")
        positive_day = rentals.reshape(-1,24).sum(axis=1) > 0
        for d in np.flatnonzero(~positive_day):
            gaps.append((station,HOURS[24*d].date(),"하루 전체에 유효한 출발 기록 없음"))
        feats = past_counts(dep, available)
        # 임의 지점 및 기간 경계에서 이벤트 단위로 다시 세어 시차를 검증한다.
        probes = [168, 24*31, 8760, int((TRAIN_END-HOURS[0]).total_seconds()/3600),
                  int((TEST_START-HOURS[0]).total_seconds()/3600), len(HOURS)-1]
        for t in probes:
            for lag in LAGS:
                assert feats[f"rentals_lag_{lag}h"][t] == ((dep==t-lag)&(available<=t)).sum()
            for win in WINDOWS:
                direct = ((dep>=t-win)&(dep<t)&(available<=t)).sum()/win
                assert np.isclose(feats[f"rentals_mean_{win}h"][t],direct)
        for lag in LAGS:
            key = f"rentals_lag_{lag}h"
            feats[key] = pd.array(feats[key], dtype="Int32")
        target_frame = pd.DataFrame({"station_id":station, "target_rentals":rentals}, index=HOURS)
        panels = pd.concat([target_frame, calendar, pd.DataFrame(feats,index=HOURS)], axis=1)
        frames.append(panels)
    panel = pd.concat(frames).rename_axis("target_time").reset_index()
    if panel.duplicated(["station_id","target_time"]).any() or len(panel)!=len(ids)*len(HOURS):
        raise ValueError("대여소 또는 시간축 중복")
    if panel.target_rentals.sum() != counts.sum() or not (panel.weather_time < panel.target_time).all():
        raise ValueError("건수 또는 기상 시차 검증 실패")
    train_counts = panel[panel.split=="train"].groupby("station_id").target_rentals.sum()
    if not np.array_equal(train_counts.loc[ids].to_numpy(), stations.train_departures.to_numpy()):
        raise ValueError("학습 기간 집계 불일치")
    monthly = pd.read_csv(cache/"monthly.csv")
    sample_column = f"표본 {len(ids)}곳 대여 건수"
    monthly[sample_column] = panel.groupby(panel.target_time.dt.strftime("%Y-%m")).target_rentals.sum().reindex(monthly["월"]).to_numpy()
    assert monthly[sample_column].sum() == counts.sum()
    # 최종 파일은 모든 검사가 끝난 후 교체한다.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        panel.to_csv(staging/"station_hour.csv.gz", index=False, encoding="utf-8",
                     float_format="%.8f", compression={"method":"gzip","compresslevel":4})
        stations.to_csv(staging/"선정 대여소.csv", index=False, encoding="utf-8-sig")
        monthly.to_csv(staging/"월별 집계 검증.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(gaps, columns=["station_id","날짜","사유"]).to_csv(staging/"대여기록 공백일.csv", index=False, encoding="utf-8-sig")
        for f in staging.iterdir():
            f.replace(OUT/f.name)
    print("선정 가능 대여소",pool_size,"표본",len(ids), "행",len(panel),"대여건수",int(panel.target_rentals.sum()),"유효 대여 0건인 대여소·날짜",len(gaps))
    print(panel.groupby("split").target_rentals.agg(["size","count","sum"]).to_string())
    print(monthly.sum(numeric_only=True).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cache", type=Path, help="이미 집계한 임시 폴더를 재사용한다.")
    args = parser.parse_args()
    if args.use_cache:
        build(args.use_cache)
    else:
        with tempfile.TemporaryDirectory(prefix="bike_prep_") as folder:
            cache = Path(folder)
            run_aggregation(cache)
            build(cache)

