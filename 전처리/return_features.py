"""유효 대여 원본에서 50개 목적지의 직전 1·3시간 반납 건수를 집계한다."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "전처리" / "대여소별_직전_반납.csv.gz"
HOURS = pd.date_range("2023-01-01", "2025-01-01", freq="h", inclusive="left")
START = HOURS[0].value // 10**9


def read_month(stream, month):
    header = next(csv.reader([stream.readline().decode("cp949")]))
    fields = {"자전거번호": pa.binary(), "대여일시": pa.timestamp("s"),
              "대여 대여소번호": pa.int32(), "반납일시": pa.timestamp("s"), "반납대여소번호": pa.int32()}
    names = [str(i) for i in range(len(header))]
    types = {str(header.index(k)): v for k, v in fields.items()}
    reader = arrow_csv.open_csv(stream,
        read_options=arrow_csv.ReadOptions(column_names=names, block_size=8*1024**2),
        parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
        convert_options=arrow_csv.ConvertOptions(column_types=types, include_columns=list(types),
                                                 null_values=["", "NULL", r"\N"]))
    parts, bike_lookup = [], {}
    start = pd.Timestamp(month).value // 10**9
    seconds_in_month = pd.Timestamp(month).days_in_month * 86400
    for batch in reader:
        bikes, dep, orig, end, dest = batch.columns
        if any(col.null_count for col in [bikes, dep, orig, end]):
            raise ValueError(f"{month} 필수 열 결측")
        codes = bikes.dictionary_encode()
        lookup = np.array([bike_lookup.setdefault(b, len(bike_lookup)) for b in codes.dictionary.to_pylist()])
        t = dep.cast(pa.int64()).to_numpy() - start
        if t.min() < 0 or t.max() >= seconds_in_month:
            raise ValueError(f"{month} 대여월 불일치")
        key = lookup[codes.indices.to_numpy()] * seconds_in_month + t
        returned = end.cast(pa.int64()).to_numpy() - START
        parts.append(pd.DataFrame({"key": key, "origin": orig.to_numpy(),
            "dest": dest.fill_null(-1).to_numpy(), "end": returned.astype("int32"),
            "valid_time": returned >= t + (start - START)}))
    return pd.concat(parts, ignore_index=True)


def selected_return_events(records, station_ids):
    # 같은 자전거·대여시각을 먼저 서울 전체에서 대조한 뒤 목적지를 거른다.
    duplicate = records.key.duplicated(keep=False)
    repeated = records.loc[duplicate & records.valid_time].copy()
    ambiguous_origin = records.loc[duplicate].groupby("key").origin.nunique()
    repeated = repeated.loc[~repeated.key.isin(ambiguous_origin.index[ambiguous_origin.gt(1)])]
    if len(repeated):
        repeated = repeated.loc[repeated.end.eq(repeated.groupby("key").end.transform("max"))]
        repeated = repeated.loc[repeated.groupby("key").dest.transform("nunique").eq(1)].drop_duplicates("key")
    columns = ["origin", "dest", "end"]
    singles = records.loc[~duplicate & records.valid_time & records.dest.isin(station_ids), columns]
    return pd.concat([singles, repeated.loc[repeated.dest.isin(station_ids), columns]], ignore_index=True)


def aggregate_month(records, station_ids):
    hits = selected_return_events(records, station_ids)
    hours = hits.end.to_numpy() // 3600
    inside = (hours >= 0) & (hours < len(HOURS))
    indices = station_ids.get_indexer(hits.dest.to_numpy()[inside])
    matrix = np.bincount(hours[inside] * len(station_ids) + indices,
        minlength=len(HOURS)*len(station_ids)).reshape(len(HOURS), len(station_ids))
    assert matrix.sum() == inside.sum()
    return matrix, int((~hits.origin.isin(station_ids) & inside).sum())


def build_return_features():
    sources = ROOT / "분석용 데이터"
    expected = pd.read_csv(sources / "월별 제외 건수.csv").set_index("월")["분석용 대여 건수"]
    station_ids = pd.Index(pd.read_csv(ROOT / "전처리" / "선정 대여소.csv").station_id, name="station_id")
    assert len(station_ids) == station_ids.nunique() == 50
    totals = np.zeros((len(HOURS), len(station_ids)), dtype="int64")
    months, outside = [], 0
    for archive in sorted(sources.glob("대여이력*.zip")):
        with ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                month = Path(member).stem[-7:]
                with z.open(member) as stream:
                    records = read_month(stream, month)
                if len(records) != expected.loc[month]:
                    raise ValueError(f"{month} 원본 행 수 불일치")
                counts, external = aggregate_month(records, station_ids)
                totals += counts
                outside += external
                months.append(month)
                del records
                print(f"{month} 대상 대여소 반납 {counts.sum():,}건", flush=True)
    assert months == list(pd.period_range("2023-01","2024-12",freq="M").astype(str))
    assert 0 < outside <= totals.sum()
    hourly = pd.DataFrame(totals, index=HOURS, columns=station_ids)
    # t행의 값은 [t-1h,t), [t-3h,t). 정각의 반납은 다음 예측부터 사용한다.
    last = hourly.shift(1)
    last_three = last.rolling(3, min_periods=3).sum()
    result = pd.DataFrame({"target_time": HOURS.repeat(len(station_ids)),
        "station_id": np.tile(station_ids,len(HOURS)), "returns_prev_1h": last.to_numpy().ravel(),
        "returns_prev_3h": last_three.to_numpy().ravel()})
    assert result.returns_prev_1h.isna().sum() == 50 and result.returns_prev_3h.isna().sum() == 150
    np.testing.assert_array_equal(last_three.iloc[3:].to_numpy(),
        totals[:-3] + totals[1:-2] + totals[2:-1])
    with NamedTemporaryFile(dir=OUT.parent, suffix=".gz", delete=False) as f:
        temp = Path(f.name)
    try:
        result.to_csv(temp, index=False, compression="gzip", float_format="%.0f")
        temp.replace(OUT)
    finally:
        temp.unlink(missing_ok=True)
    print(f"전체 반납 {totals.sum():,}건 중 표본 밖 출발 {outside:,}건", flush=True)


if __name__ == "__main__":
    build_return_features()
