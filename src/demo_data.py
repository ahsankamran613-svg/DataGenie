"""Built-in realistic messy Retail & E-Commerce dataset generator for DataGenie.

Generates ~668 raw rows (~650 unique orders + 18 intentional duplicate rows) across
6 months (May 2026 – October 2026) with:
1. Intentional data quality issues:
   - 9 unstandardized column headers with spaces and symbols
   - 18 exact duplicate rows
   - 74 missing values across Discount (%) and Shipping Delay (Days)
   - Currency/percentage strings ("$1,420.50", "15%"), extra whitespace, mixed casing (" north ", "NORTH")
   - 3 mixed date formats ("YYYY-MM-DD", "YYYY/MM/DD", "DD-MM-YYYY")
2. Embedded business narrative for AI Root-Cause Analysis:
   - ~32% revenue drop in the North region in October 2026 vs September 2026 ($184K -> $125K)
   - Driven primarily by Electronics supply stockouts (~41% units drop), shipping delays
     spiking from ~2 days to ~9 days, and discontinued promo discounts on Oct 1.
   - 3 clear daily revenue anomalies for IsolationForest + Rolling Z-score detection.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_DEMO_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "messy_regional_sales.csv"

CATEGORY_SPECS: dict[str, dict[str, float]] = {
    "Electronics": {"unit_price": 185.0, "base_margin": 0.21},
    "Apparel": {"unit_price": 68.0, "base_margin": 0.36},
    "Home & Kitchen": {"unit_price": 112.0, "base_margin": 0.28},
    "Beauty & Health": {"unit_price": 54.0, "base_margin": 0.43},
}

REGION_WEIGHTS: dict[str, float] = {
    "North": 0.30,
    "South": 0.25,
    "East": 0.24,
    "West": 0.21,
}


def _messy_date_string(dt: pd.Timestamp, idx: int) -> str:
    """Format a date into one of three realistic mixed formats."""
    mode = idx % 5
    if mode == 0:
        return dt.strftime("%Y/%m/%d")
    if mode == 1:
        return dt.strftime("%d-%m-%Y")
    return dt.strftime("%Y-%m-%d")


def _messy_region_string(region: str, idx: int) -> str:
    """Inject realistic whitespace and casing inconsistencies into region labels."""
    variant = idx % 7
    if variant == 0:
        return f" {region.lower()} "
    if variant == 1:
        return region.upper()
    if variant == 2:
        return f"{region} "
    return region


def _messy_category_string(category: str, idx: int) -> str:
    """Inject subtle casing/whitespace inconsistencies into product category names."""
    variant = idx % 8
    if variant == 0:
        return f" {category.lower()} "
    if variant == 1:
        return f"{category}  "
    return category


def generate_messy_sales_dataframe(seed: int = 42) -> pd.DataFrame:
    """Generate the deterministic ~668-row messy regional retail sales DataFrame."""
    rng = np.random.default_rng(seed)

    # 650 unique base orders across 6 months (May–Oct 2026)
    # Allocate ~108 orders per month (647 regular + 3 explicit anomaly orders = 650)
    months = [
        ("2026-05-01", "2026-05-31", 107),
        ("2026-06-01", "2026-06-30", 108),
        ("2026-07-01", "2026-07-31", 108),
        ("2026-08-01", "2026-08-31", 108),
        ("2026-09-01", "2026-09-30", 108),
        ("2026-10-01", "2026-10-31", 108),
    ]

    regions = list(REGION_WEIGHTS.keys())
    region_probs = list(REGION_WEIGHTS.values())
    categories = list(CATEGORY_SPECS.keys())
    cat_probs = [0.36, 0.24, 0.22, 0.18]

    records: list[dict[str, object]] = []
    order_counter = 10001

    for m_idx, (start_str, end_str, count) in enumerate(months):
        start_dt = pd.Timestamp(start_str)
        days_in_month = (pd.Timestamp(end_str) - start_dt).days + 1
        is_october = m_idx == 5
        is_september = m_idx == 4

        for i in range(count):
            day_offset = int(rng.integers(0, days_in_month))
            # Keep Oct 14 clear for our explicit outage anomaly
            if is_october and day_offset == 13:
                day_offset = 14
            order_dt = start_dt + pd.Timedelta(days=day_offset)

            region = str(rng.choice(regions, p=region_probs))
            category = str(rng.choice(categories, p=cat_probs))
            spec = CATEGORY_SPECS[category]

            # Baseline units & seasonal growth for non-North regions
            growth_factor = 1.0 + (0.025 * m_idx)
            base_units = rng.integers(8, 22)

            if region == "North":
                if is_september:
                    # Strong September performance in North (~$184K target)
                    if category == "Electronics":
                        units = int(round(base_units * 1.48))
                    else:
                        units = int(round(base_units * 1.25))
                    discount = float(rng.choice([10.0, 12.0, 15.0, 18.0]))
                    shipping_delay = float(np.round(rng.uniform(1.2, 2.8), 1))
                elif is_october:
                    # October North drop (~$125K target, ~32% decline vs Sep):
                    # Electronics stockouts + shipping delay spike + promo ended Oct 1
                    if category == "Electronics":
                        units = max(2, int(round(base_units * 0.62)))
                        shipping_delay = float(np.round(rng.uniform(7.8, 10.6), 1))
                    else:
                        units = max(4, int(round(base_units * 1.02)))
                        shipping_delay = float(np.round(rng.uniform(3.0, 5.0), 1))
                    discount = 0.0
                else:
                    units = int(round(base_units * 1.22))
                    discount = float(rng.choice([8.0, 10.0, 12.0, 15.0]))
                    shipping_delay = float(np.round(rng.uniform(1.5, 2.8), 1))
            else:
                units = int(round(base_units * growth_factor))
                discount = float(rng.choice([0.0, 5.0, 10.0, 12.0, 15.0]))
                shipping_delay = float(np.round(rng.uniform(1.5, 3.6), 1))

            unit_price = spec["unit_price"] * float(rng.uniform(0.93, 1.07))
            gross = units * unit_price
            net_revenue = round(gross * (1.0 - discount / 100.0), 2)

            effective_margin = spec["base_margin"] - (discount * 0.0035) + float(rng.uniform(-0.02, 0.02))
            effective_margin = float(np.clip(effective_margin, 0.08, 0.55))
            profit = round(net_revenue * effective_margin, 2)

            records.append(
                {
                    "order_id": f"ORD-{order_counter}",
                    "dt": order_dt,
                    "region": region,
                    "category": category,
                    "units_sold": units,
                    "discount_pct": discount,
                    "shipping_delay": shipping_delay,
                    "revenue": net_revenue,
                    "profit": profit,
                }
            )
            order_counter += 1

    # Calibrate North September (~$184,200) and North October (~$125,100) for an exact ~32.1% drop
    df_clean_base = pd.DataFrame(records)
    sep_mask = (df_clean_base["region"] == "North") & (
        df_clean_base["dt"].between("2026-09-01", "2026-09-30")
    )
    oct_mask = (df_clean_base["region"] == "North") & (
        df_clean_base["dt"].between("2026-10-01", "2026-10-31")
    )

    sep_sum = float(df_clean_base.loc[sep_mask, "revenue"].sum())
    oct_sum = float(df_clean_base.loc[oct_mask, "revenue"].sum())

    if sep_sum > 0:
        sep_scale = 184200.0 / sep_sum
        df_clean_base.loc[sep_mask, "revenue"] = (
            df_clean_base.loc[sep_mask, "revenue"] * sep_scale
        ).round(2)
        df_clean_base.loc[sep_mask, "profit"] = (
            df_clean_base.loc[sep_mask, "profit"] * sep_scale
        ).round(2)

    if oct_sum > 0:
        oct_scale = 124900.0 / oct_sum
        df_clean_base.loc[oct_mask, "revenue"] = (
            df_clean_base.loc[oct_mask, "revenue"] * oct_scale
        ).round(2)
        df_clean_base.loc[oct_mask, "profit"] = (
            df_clean_base.loc[oct_mask, "profit"] * oct_scale
        ).round(2)

    # Inject 3 explicit daily anomalies (bringing total unique orders to 650)
    anomaly_rows = [
        {
            "order_id": f"ORD-{order_counter}",
            "dt": pd.Timestamp("2026-06-18"),
            "region": "East",
            "category": "Electronics",
            "units_sold": 265,
            "discount_pct": 15.0,
            "shipping_delay": 2.1,
            "revenue": 42850.00,
            "profit": 8140.50,
        },
        {
            "order_id": f"ORD-{order_counter + 1}",
            "dt": pd.Timestamp("2026-08-24"),
            "region": "West",
            "category": "Home & Kitchen",
            "units_sold": 390,
            "discount_pct": 10.0,
            "shipping_delay": 2.4,
            "revenue": 39640.00,
            "profit": 10702.80,
        },
        {
            "order_id": f"ORD-{order_counter + 2}",
            "dt": pd.Timestamp("2026-10-14"),
            "region": "South",
            "category": "Beauty & Health",
            "units_sold": 1,
            "discount_pct": 0.0,
            "shipping_delay": 11.5,
            "revenue": 48.00,
            "profit": 19.20,
        },
    ]
    df_clean_base = pd.concat([df_clean_base, pd.DataFrame(anomaly_rows)], ignore_index=True)

    # Convert clean base records into intentionally messy raw strings (9 dirty headers)
    messy_rows: list[dict[str, object]] = []
    for idx, row in df_clean_base.iterrows():
        i = int(idx)
        rev_val = float(row["revenue"])
        prof_val = float(row["profit"])
        disc_val = float(row["discount_pct"])

        rev_str = f"${rev_val:,.2f}" if i % 4 != 0 else f" ${rev_val:,.2f} "
        prof_str = f"${prof_val:,.2f}" if i % 5 != 0 else f"{prof_val:.2f}"
        disc_str: object = f"{disc_val:.0f}%" if i % 3 != 0 else f" {disc_val:.1f}% "

        messy_rows.append(
            {
                "Order ID": row["order_id"],
                "Order Date": _messy_date_string(pd.Timestamp(row["dt"]), i),
                "Region ": _messy_region_string(str(row["region"]), i),
                "Product Category": _messy_category_string(str(row["category"]), i),
                "Units Sold": int(row["units_sold"]),
                "Discount (%)": disc_str,
                "Shipping Delay (Days)": float(row["shipping_delay"]),
                "Revenue ($)": rev_str,
                "Profit ($)": prof_str,
            }
        )

    messy_df = pd.DataFrame(messy_rows)

    # Inject 74 missing values across non-North-Oct critical rows (40 in Discount, 34 in Shipping Delay)
    # Using deterministic non-overlapping indices on unique rows BEFORE duplicating
    candidate_indices = messy_df.index[:-3].to_numpy()
    null_discount_idx = rng.choice(candidate_indices, size=40, replace=False)
    remaining_idx = np.setdiff1d(candidate_indices, null_discount_idx)
    null_delay_idx = rng.choice(remaining_idx, size=34, replace=False)

    messy_df.loc[null_discount_idx, "Discount (%)"] = np.nan
    messy_df.loc[null_delay_idx, "Shipping Delay (Days)"] = np.nan

    # Add 18 exact duplicate rows (picking rows without missing values so missing_before stays 74!)
    complete_indices = np.setdiff1d(
        candidate_indices, np.union1d(null_discount_idx, null_delay_idx)
    )
    dup_indices = rng.choice(complete_indices, size=18, replace=False)
    duplicates_df = messy_df.loc[dup_indices].copy()

    final_messy_df = pd.concat([messy_df, duplicates_df], ignore_index=True)
    # Shuffle deterministically so duplicates are scattered naturally throughout the CSV
    shuffled_order = rng.permutation(len(final_messy_df))
    final_messy_df = final_messy_df.iloc[shuffled_order].reset_index(drop=True)

    return final_messy_df


def load_or_create_demo_csv(csv_path: str | Path | None = None, force_rebuild: bool = False) -> pd.DataFrame:
    """Load the messy regional sales demo CSV, generating and saving it on first run."""
    target_path = Path(csv_path) if csv_path is not None else DEFAULT_DEMO_CSV_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if force_rebuild or not target_path.exists():
        df = generate_messy_sales_dataframe(seed=42)
        df.to_csv(target_path, index=False)
        return df

    return pd.read_csv(target_path)
