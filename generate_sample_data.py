"""
ChatGPT Enterprise 통계 내보내기 포맷의 샘플 POC 데이터를 생성합니다.

컬럼: date_partition, email, 사업부, 부서, usage_type, usage_credit, usage_quantity, usage_unit
"""
import argparse
import csv
import random
from datetime import date, timedelta


DIVISIONS = ["개발본부", "영업본부", "마케팅본부", "경영지원본부", "연구소"]
DEPARTMENTS = {
    "개발본부": ["백엔드개발팀", "프론트엔드개발팀", "인프라팀", "QA팀"],
    "영업본부": ["영업1팀", "영업2팀", "고객성공팀"],
    "마케팅본부": ["디지털마케팅팀", "브랜드팀", "콘텐츠팀"],
    "경영지원본부": ["인사팀", "재무팀", "법무팀"],
    "연구소": ["AI연구팀", "제품연구팀"],
}

# (usage_type, usage_unit, credit_per_unit_range, qty_range)
USAGE_PROFILES = [
    ("gpt-4o",          "tokens",     (0.002,  0.015),  (500,   8000)),
    ("gpt-4o-mini",     "tokens",     (0.0002, 0.002),  (1000, 20000)),
    ("o1",              "tokens",     (0.01,   0.05),   (200,   3000)),
    ("dall-e-3",        "counts",     (0.04,   0.08),   (1,     10)),
    ("whisper",         "duration_s", (0.001,  0.006),  (30,    600)),
]

# User activity: fraction of days they're active (Beta distribution parameters)
USER_ACTIVITY_PROFILES = [
    # (fraction_of_users, mean_active_days_ratio, intensity_multiplier)
    (0.20, 0.65, 5.0),   # Heavy: active most days, high intensity
    (0.40, 0.35, 1.5),   # Medium
    (0.40, 0.12, 0.4),   # Light
]


def generate(users: int, days: int, seed: int, output: str):
    rng = random.Random(seed)

    start_date = date(2025, 1, 1)
    dates = [start_date + timedelta(d) for d in range(days)]

    # Assign user profiles
    user_info = []
    for i in range(users):
        division = rng.choice(DIVISIONS)
        department = rng.choice(DEPARTMENTS[division])
        email = f"user{i+1:03d}@company.com"

        # Determine segment
        r = rng.random()
        cumulative = 0
        for frac, active_ratio, intensity in USER_ACTIVITY_PROFILES:
            cumulative += frac
            if r <= cumulative:
                break

        # Daily active probability
        active_prob = max(0.02, min(0.95, rng.gauss(active_ratio, active_ratio * 0.3)))
        user_info.append((email, division, department, active_prob, intensity))

    rows = []
    for email, division, department, active_prob, intensity in user_info:
        # Dominant usage type for this user
        dominant_type_idx = rng.choices(
            range(len(USAGE_PROFILES)),
            weights=[0.45, 0.35, 0.08, 0.07, 0.05],
        )[0]

        for day in dates:
            if rng.random() > active_prob:
                continue

            # 1-3 usage types per active day
            n_types = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            chosen = [dominant_type_idx]
            others = list(range(len(USAGE_PROFILES)))
            others.remove(dominant_type_idx)
            chosen += rng.sample(others, min(n_types - 1, len(others)))

            for idx in chosen:
                ut, unit, cr_range, qty_range = USAGE_PROFILES[idx]
                qty = rng.uniform(*qty_range) * intensity
                credit_per_unit = rng.uniform(*cr_range)
                credit = qty * credit_per_unit

                rows.append({
                    "date_partition": day.isoformat(),
                    "email": email,
                    "사업부": division,
                    "부서": department,
                    "usage_type": ut,
                    "usage_credit": round(credit, 6),
                    "usage_quantity": round(qty, 2),
                    "usage_unit": unit,
                })

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "date_partition", "email", "사업부", "부서",
            "usage_type", "usage_credit", "usage_quantity", "usage_unit",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"샘플 데이터 생성 완료: {output}")
    print(f"  사용자: {users}명, 기간: {days}일, 레코드: {len(rows):,}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POC 샘플 CSV 생성기")
    parser.add_argument("--users", type=int, default=100, help="사용자 수 (기본: 100)")
    parser.add_argument("--days", type=int, default=60, help="POC 기간 일수 (기본: 60)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본: 42)")
    parser.add_argument("-o", "--output", default="sample_poc_data.csv", help="출력 파일명")
    args = parser.parse_args()

    generate(args.users, args.days, args.seed, args.output)
