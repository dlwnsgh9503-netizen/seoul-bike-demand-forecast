"""2024년 12월 완전연결 군집의 시간별 반납 건수를 목적지 기준으로 집계한다."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd

from return_features import HOURS, ROOT, read_month, selected_return_events


OUT = ROOT / "전처리" / "1021개 군집 시간별 반납.npz"


def build_cluster_returns():
    inputs = ROOT / "분석용 데이터"
    mapping = pd.read_csv(ROOT / "따릉이 대여소 비교/서울 전체 직선거리 500m 군집 배정.csv")
    group = mapping.loc[mapping["기준월"] == "2024-12"].set_index("대여소번호")["완전연결 군집번호"].sort_index()
    ids = pd.Index(sorted(group.unique()))
    assert group.index.is_unique and len(group) == 2766 and len(ids) == 1021
    expected = pd.read_csv(inputs / "월별 제외 건수.csv").set_index("월")["분석용 대여 건수"]
    sample = pd.Index(pd.read_csv(ROOT / "전처리/선정 대여소.csv").station_id).intersection(group.index)
    audit = np.zeros((len(HOURS), len(sample)), dtype=np.int32)
    counts = np.zeros((len(HOURS), len(ids)), dtype=np.int32)
    months, assigned = [], 0
    for archive in sorted(inputs.glob("대여이력*.zip")):
        with ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                month = Path(member).stem[-7:]
                with z.open(member) as stream:
                    records = read_month(stream, month)
                assert len(records) == expected.loc[month]
                hits = selected_return_events(records, group.index)
                del records
                h = hits.end.to_numpy() // 3600
                inside = (h >= 0) & (h < len(HOURS))
                dest = hits.dest.to_numpy()[inside]
                h = h[inside]
                clusters = ids.get_indexer(group.loc[dest])
                assert (clusters >= 0).all()
                np.add.at(counts, (h, clusters), 1)
                sample_pos = sample.get_indexer(dest)
                keep = sample_pos >= 0
                np.add.at(audit, (h[keep], sample_pos[keep]), 1)
                assigned += len(dest)
                months.append(month)
                print(f"{month} 군집에 반납 {len(dest):,}건", flush=True)
    assert months == list(pd.period_range("2023-01", "2024-12", freq="M").astype(str))
    assert counts.sum() == assigned
    reference = pd.read_csv(inputs / "대여소별_직전_반납.csv.gz", parse_dates=["target_time"])
    prev_hour = reference.pivot(index="target_time", columns="station_id", values="returns_prev_1h")
    np.testing.assert_array_equal(audit[:-1], prev_hour.loc[HOURS[1:], sample].to_numpy())
    with NamedTemporaryFile(dir=OUT.parent, suffix=".npz", delete=False) as temp:
        path = Path(temp.name)
    try:
        np.savez_compressed(path, counts=counts, ids=ids.to_numpy(), station_ids=group.index.to_numpy(),
                            station_groups=group.to_numpy())
        path.replace(OUT)
    finally:
        path.unlink(missing_ok=True)
    print(f"반납 {assigned:,}건, 기존 표본 {len(sample)}개소 시간별 집계 대조 완료", flush=True)


if __name__ == "__main__":
    build_cluster_returns()
