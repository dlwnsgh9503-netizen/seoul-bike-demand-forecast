"""서울 따릉이 월별 ZIP에서 대여소별 시간 집계와 반납 확인 시각을 준비한다."""
import csv
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "분석용 데이터"
START = np.datetime64("2023-01-01T00:00:00", "s").astype("int64")
COLUMNS = ["자전거번호", "대여일시", "대여 대여소번호", "반납일시"]


def seconds(values):
    dates = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if dates.isna().any():
        raise ValueError("시각 결측")
    return dates.to_numpy(dtype="datetime64[s]").astype("int64") - START


def candidate_records(archive, member, indices, expected):
    """CSV의 논리 행 번호에 해당하는 후보만 읽는다. 인용된 줄바꿈도 처리한다."""
    found = {}
    position = -1  # 헤더
    with ZipFile(archive) as z, z.open(member) as stream:
        while True:
            block = stream.read(2**20)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += stream.readline()
            size = block.count(b"\n") + int(not block.endswith(b"\n"))
            first, last = np.searchsorted(indices, [position, position + size])
            if first < last:
                lines = block.split(b"\n")
                for i in indices[first:last]:
                    found[int(i)] = lines[int(i)-position]
            position += size
    if position == expected and len(found) == len(indices):
        return {i: tuple(next(csv.reader([s.decode("cp949")]))) for i,s in found.items()}
    wanted = set(indices)
    with ZipFile(archive) as z, z.open(member) as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding="cp949", newline=""))
        next(reader)
        return {i: tuple(row) for i,row in enumerate(reader) if i in wanted}


def reconcile(archive, member, keys, binary, matrix, offset):
    order = np.argsort(keys, kind="stable")
    same = keys[order[:-1]] == keys[order[1:]]
    candidates = np.sort(order[np.r_[same, False] | np.r_[False, same]])
    if not len(candidates):
        return [0]*5
    records = candidate_records(archive, member, candidates, len(keys))
    groups = {}
    for i in candidates:
        row = records[i]
        if len(row)!=17:
            raise ValueError("중복 후보 CSV 구조 오류")
        groups.setdefault((row[0], row[1]), []).append(int(i))
    if any(len(v)<2 for v in groups.values()):
        raise ValueError("원본 행과 중복 키 불일치")
    events = np.memmap(binary, dtype="<i4", mode="r+").reshape(-1,3)
    removed = []
    identical = variants = conflicts = conflict_rows = 0
    for group in groups.values():
        unique = {}
        for i in group:
            unique.setdefault(records[i],i)
        identical += len(group)-len(unique)
        if len({int(row[2]) for row in unique}) > 1:
            # 출발 대여소가 서로 달라 어느 대여소의 실적인지 결정할 수 없다.
            conflicts += 1
            conflict_rows += len(unique)
            removed.extend(group)
        else:
            variants += len(unique)-1
            # 동일 출발의 반납 시각이 다르면 가장 늦은 확인 시각을 사용한다.
            keep = max(unique.values(), key=lambda i: events[i,2])
            removed.extend(i for i in group if i != keep)
    for i in removed:
        station, hour, _ = events[i]
        matrix[int(station)][int(hour)-offset] -= 1
    events[removed,0] = 0  # 원본 ZIP과 행 번호 대응을 보존하는 임시 플래그
    events.flush()
    del events
    assert identical + variants + conflict_rows == len(removed)
    return [len(candidates)-len(groups),identical,variants,conflict_rows,conflicts]


def aggregate_month(job):
    archive, member, month, expected, cache = job
    start = pd.Timestamp(month)
    offset = (start - pd.Timestamp("2023-01-01")).days * 24
    nh = start.days_in_month * 24
    matrix = {}
    bike_codes = {}
    keys = []
    row_count = 0
    early_returns = 0
    with ZipFile(archive) as z, z.open(member) as stream, (cache / (month + ".bin")).open("wb") as binary:
        for chunk in pd.read_csv(stream, encoding="cp949", usecols=COLUMNS,
                                 dtype={"대여 대여소번호": "int32", "자전거번호": "str"}, chunksize=200_000):
            departure = seconds(chunk["대여일시"])
            returned = seconds(chunk["반납일시"])
            local_hour = departure // 3600 - offset
            ids = chunk["대여 대여소번호"].to_numpy()
            if local_hour.min() < 0 or local_hour.max() >= nh or ids.min() <= 0:
                raise ValueError(f"{month} 파일의 대여월 또는 대여소번호 오류")
            early_returns += int((returned < departure).sum())
            # 반납시각 이상 행은 확정된 과거 실적으로 쓰지 않는다.
            available = np.where(returned >= departure, returned // 3600 + 1, np.iinfo(np.int32).max)
            temp = np.column_stack((ids, local_hour + offset, available)).astype("<i4")
            temp.tofile(binary)
            codes, stations = pd.factorize(ids)
            counts = np.bincount(codes * nh + local_hour,
                                 minlength=len(stations) * nh).reshape(-1, nh)
            for station, row in zip(stations, counts):
                n = int(station)
                if n not in matrix:
                    matrix[n] = np.zeros(nh, dtype="int32")
                matrix[n] += row.astype("int32")
            bc, bikes = pd.factorize(chunk["자전거번호"])
            if (bc < 0).any():
                raise ValueError("자전거번호 결측")
            lookup = np.array([bike_codes.setdefault(b, len(bike_codes)) for b in bikes], dtype="int64")
            keys.append(lookup[bc] * (nh * 3600) + (departure - offset * 3600))
            row_count += len(chunk)
    if row_count != expected or sum(int(x.sum()) for x in matrix.values()) != expected:
        raise ValueError(f"{month} 원천 집계 불일치")
    dup, exact, same_start, ambiguous, groups = reconcile(
        archive, member, np.concatenate(keys), cache/(month+".bin"), matrix, offset)
    ids = np.array(sorted(matrix), dtype="int32")
    counts = np.stack([matrix[i] for i in ids])
    kept = row_count-exact-same_start-ambiguous
    if counts.min()<0 or counts.sum()!=kept:
        raise ValueError(f"{month} 중복 처리 후 합계 불일치")
    np.savez_compressed(cache / (month + ".npz"), ids=ids, counts=counts)
    print(month, row_count, "원천 /", kept, "반영, 완전중복/동일출발/출발충돌", exact, same_start, ambiguous, flush=True)
    return (month, row_count, dup, exact, same_start, ambiguous, groups, kept, early_returns)


def run_aggregation(cache):
    cache.mkdir(parents=True, exist_ok=True)
    expected = pd.read_csv(INPUT / "월별 제외 건수.csv").set_index("월")["분석용 대여 건수"]
    jobs = []
    for archive in sorted(INPUT.glob("대여이력*.zip")):
        with ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                month = Path(member).stem[-7:]
                jobs.append((archive, member, month, int(expected.loc[month]), cache))
    if sorted(j[2] for j in jobs) != list(pd.period_range("2023-01", "2024-12", freq="M").astype(str)):
        raise ValueError("월별 입력 24개를 확인하세요.")
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(aggregate_month, jobs))
    pd.DataFrame(rows, columns=["월", "필터 후 원천 행", "자전거번호와 대여일시 중복 추가행", "완전 동일 제외 행", "동일 출발 중복 제외 행", "출발 대여소 불일치 제외 행", "출발 대여소 불일치 그룹", "집계 대여 건수", "대여보다 이른 반납시각"]).to_csv(
        cache / "monthly.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    import sys
    run_aggregation(Path(sys.argv[1]))

