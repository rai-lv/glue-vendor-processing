#!/usr/bin/env python3
# Glue 5.0 / Spark / Python 3 script
#
# Job name (example):
#   vendorInputProcessing_VENDOR_mappingMethodTraining
#
# High-level behaviour:
#   - Reads a "oneVendor_to_onePim_match" JSON for a given vendor from prepared_input_key.
#   - Reads/writes the global Stable Training Dataset
#       s3://<INPUT_BUCKET>/canonical_mappings/Category_Mapping_StableTrainingDataset.json
#   - Rebuilds the "training subset" in-memory (and writes the JSON mirror
#     Category_Mapping_StableTrainingDataset_training.json only for transparency).
#   - Derives rule proposals based on KEYWORD statistics:
#       * K1 (existing): contains_any, only when hard_outside_count == 0.
#       * K2 (new): contains_any_exclude_any, only when K1 fails because
#         hard_outside_count > 0, using excludes that exist only in outside
#         categories and never in the inside category, and that fully cover all
#         outside occurrences.
#   - Reads the latest Category_Mapping_Reference_<timestamp>.json, updates
#     mapping_methods (vendor_mappings remain unchanged), writes a new
#     timestamped reference file and a change log JSON next to it.
#
# Safety principles:
#   - No removal or rewriting of existing business logic for K1.
#   - All new behaviour for K2 is additive and clearly separated in STEP D5.

import json
import sys
import datetime
from typing import Any, Dict, List, Optional, Tuple

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql import types as T

import boto3


# ========================================================================
# Logging helpers
# ========================================================================


def log_info(logger, msg: str) -> None:
    try:
        logger.info(msg)
    except Exception:
        # Fallback if logger is not fully available (e.g. early in init)
        print(f"[INFO] {msg}")


def log_warning(logger, msg: str) -> None:
    try:
        logger.warn(msg)
    except Exception:
        print(f"[WARN] {msg}")


def log_error(logger, msg: str) -> None:
    try:
        logger.error(msg)
    except Exception:
        print(f"[ERROR] {msg}")


# ========================================================================
# S3 helpers
# ========================================================================


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.NoSuchKey:
        return False
    except Exception:
        return False


def read_json_from_s3(s3_client, bucket: str, key: str) -> Any:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def write_json_to_s3(
    s3_client,
    bucket: str,
    key: str,
    data: Any,
    indent: Optional[int] = 2,
    logger=None,
) -> None:
    body = json.dumps(data, indent=indent, ensure_ascii=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
    if logger is not None:
        log_info(logger, f"Wrote JSON to s3://{bucket}/{key} (len={len(body)})")
    else:
        print(f"[INFO] Wrote JSON to s3://{bucket}/{key} (len={len(body)})")


# ========================================================================
# Normalisation helpers for training records
# ========================================================================


def normalise_keywords(raw_keywords: Any) -> List[str]:
    """
    Normalise "keywords" field into a list of non-empty strings.
    """
    if raw_keywords is None:
        return []

    if isinstance(raw_keywords, list):
        out = []
        for v in raw_keywords:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out

    # Single scalar
    s = str(raw_keywords).strip()
    return [s] if s else []


def normalise_training_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalises a single training record to a canonical structure.
    """

    return {
        "pim_category_id": rec.get("pim_category_id"),
        "pim_category_name": rec.get("pim_category_name"),
        "pim_category_path": rec.get("pim_category_path"),
        "vendor_short_name": rec.get("vendor_short_name"),
        "vendor_category_id": rec.get("vendor_category_id"),
        "vendor_category_name": rec.get("vendor_category_name"),
        "vendor_category_path": rec.get("vendor_category_path"),
        "article_id": rec.get("article_id"),
        "description_short": rec.get("description_short"),
        "keywords": normalise_keywords(rec.get("keywords")),
        "class_codes": rec.get("class_codes") or [],
    }


# ========================================================================
# Category Mapping Reference helpers (STEP E)
# ========================================================================


def parse_reference_timestamp_from_key(key: str) -> Optional[datetime.datetime]:
    """
    Extracts timestamp from a key like:
      canonical_mappings/Category_Mapping_Reference_YYYYMMDD-HHMMSS.json
    """
    try:
        # Very simple parse: split on "Category_Mapping_Reference_"
        # then up to ".json"
        prefix = "Category_Mapping_Reference_"
        if prefix not in key:
            return None
        part = key.split(prefix, 1)[1]
        ts_str = part.split(".json", 1)[0]
        return datetime.datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except Exception:
        return None


def load_latest_category_mapping_reference(
    s3_client, bucket: str, logger=None
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Returns (key, data) for the latest Category_Mapping_Reference_*.json file
    in s3://bucket/canonical_mappings/.
    If no such file exists, returns (None, []).
    """
    prefix = "canonical_mappings/Category_Mapping_Reference_"
    log_info(
        logger,
        f"STEP E1: Scanning s3://{bucket}/canonical_mappings/ for latest Category_Mapping_Reference_* file",
    )

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix="canonical_mappings/")

    best_key = None
    best_ts: Optional[datetime.datetime] = None

    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if "Category_Mapping_Reference_" not in key:
                continue
            ts = parse_reference_timestamp_from_key(key)
            if ts is None:
                continue
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_key = key

    if best_key is None:
        log_warning(
            logger,
            "STEP E1: No Category_Mapping_Reference_* file found; starting with empty reference.",
        )
        return None, []

    log_info(
        logger,
        f"STEP E1: Latest Category_Mapping_Reference found: s3://{bucket}/{best_key} (timestamp={best_ts})",
    )
    data = read_json_from_s3(s3_client, bucket, best_key)
    if not isinstance(data, list):
        log_warning(
            logger,
            "STEP E1: Latest Category_Mapping_Reference is not a list; treating as empty.",
        )
        return best_key, []

    log_info(
        logger,
        f"STEP E1: Loaded Category_Mapping_Reference with {len(data)} PIM category entries.",
    )
    return best_key, data


def build_reference_index(
    reference_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Builds an index from pim_category_id -> reference record.
    We assume (after the user's clean-up) that there is at most one entry per
    PIM category, possibly with multiple vendor_mappings inside.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for entry in reference_data:
        cid = entry.get("pim_category_id")
        if cid is None:
            continue
        cid_str = str(cid)
        if cid_str not in index:
            index[cid_str] = entry
        else:
            # If duplicates exist, we keep the first one and ignore the rest.
            # User has already cleaned up data so this should not normally occur.
            pass
    return index


def mapping_method_signature(method: Dict[str, Any]) -> Tuple:
    """
    Create a deterministic signature for a mapping method so that we can
    compare / deduplicate rules. We do NOT touch vendor_mappings, only
    mapping_methods.
    """
    field_name = method.get("field_name")
    operator = method.get("operator")
    values_include = method.get("values_include") or []
    values_exclude = method.get("values_exclude") or []

    sig = (
        field_name,
        operator,
        tuple(values_include),
        tuple(values_exclude),
    )
    return sig


def update_mapping_methods_from_rule_proposals(
    logger,
    reference_index: Dict[str, Dict[str, Any]],
    rule_proposals: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Applies the following logic to mapping_methods per pim_category_id:
      - We DO NOT touch vendor_mappings at all.
      - For each PIM category:
          * Remove all existing mapping_methods.
          * For each rule_proposal with vendor_status in {new, supported, not_impacted}:
              - Add its mapping method (field_name, operator, values_include, values_exclude).
          * Ignore any "violated" rules.
      - Returns:
          (updated_reference_list, change_log_list)

    The change log contains entries per PIM category with:
      - pim_category_id
      - pim_category_name
      - previous_signatures (list of tuple signatures)
      - new_signatures (list of tuple signatures)
    """

    ACCEPTED_STATUSES = {"new", "supported", "not_impacted"}

    change_log: List[Dict[str, Any]] = []

    # Collect mapping methods per pim_category based on proposals
    proposals_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for rp in rule_proposals:
        stats = rp.get("stats") or {}
        status = stats.get("vendor_status")
        if status not in ACCEPTED_STATUSES:
            continue

        cid = rp.get("pim_category_id")
        if cid is None:
            continue
        cid_str = str(cid)

        method = {
            "field_name": rp.get("field_name"),
            "operator": rp.get("operator"),
            "values_include": rp.get("values_include") or [],
            "values_exclude": rp.get("values_exclude") or [],
        }

        proposals_by_cat.setdefault(cid_str, []).append(method)

    # Apply per category
    for cid_str, ref_entry in reference_index.items():
        old_methods = ref_entry.get("mapping_methods") or []
        old_sigs = [mapping_method_signature(m) for m in old_methods]

        new_methods = []
        if cid_str in proposals_by_cat:
            for m in proposals_by_cat[cid_str]:
                new_methods.append(m)
        new_sigs = [mapping_method_signature(m) for m in new_methods]

        if old_sigs != new_sigs:
            ref_entry["mapping_methods"] = new_methods
            change_log.append(
                {
                    "pim_category_id": cid_str,
                    "pim_category_name": ref_entry.get("pim_category_name"),
                    "previous_signatures": old_sigs,
                    "new_signatures": new_sigs,
                }
            )
        else:
            ref_entry["mapping_methods"] = old_methods

    # Convert index back to list
    updated_reference_list = list(reference_index.values())
    return updated_reference_list, change_log


# ========================================================================
# Main Glue job
# ========================================================================


def main():
    print("GLUE SCRIPT (mappingMethodTraining): module import started")

    # --------------------------------------------------------------------
    # Parse Glue arguments
    # --------------------------------------------------------------------
    current_step = "Argument parsing"
    logger = None

    try:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_BUCKET",
                "OUTPUT_BUCKET",
                "vendor_name",
                "prepared_input_key",
                "prepared_output_prefix",
            ],
        )
        sc = SparkContext()
        glue_context = GlueContext(sc)
        spark = glue_context.spark_session
        logger = glue_context.get_logger()

        job = Job(glue_context)
        job.init(args["JOB_NAME"], args)

        log_info(logger, "GLUE SCRIPT: Parsing Glue arguments via getResolvedOptions()")
        print("GLUE SCRIPT: Parsing Glue arguments via getResolvedOptions()")

        JOB_NAME = args["JOB_NAME"]
        INPUT_BUCKET = args["INPUT_BUCKET"]
        OUTPUT_BUCKET = args["OUTPUT_BUCKET"]
        vendor_name = args["vendor_name"]
        prepared_input_key = args["prepared_input_key"]
        prepared_output_prefix = args["prepared_output_prefix"]

        log_info(
            logger,
            f"GLUE SCRIPT: Parsed arguments: "
            f"JOB_NAME={JOB_NAME}, INPUT_BUCKET={INPUT_BUCKET}, OUTPUT_BUCKET={OUTPUT_BUCKET}, "
            f"vendor_name={vendor_name}, prepared_input_key={prepared_input_key}, "
            f"prepared_output_prefix={prepared_output_prefix}",
        )
        print(
            "GLUE SCRIPT: Parsed arguments: "
            f"JOB_NAME={JOB_NAME}, INPUT_BUCKET={INPUT_BUCKET}, OUTPUT_BUCKET={OUTPUT_BUCKET}, "
            f"vendor_name={vendor_name}, prepared_input_key={prepared_input_key}, "
            f"prepared_output_prefix={prepared_output_prefix}"
        )

        log_info(logger, "GLUE SCRIPT: Creating SparkContext and GlueContext.")
        print("GLUE SCRIPT: Creating SparkContext and GlueContext.")

        log_info(logger, "GLUE SCRIPT: Creating Glue Job and logger.")

        log_info(logger, "GLUE SCRIPT: Glue Job initialised successfully.")
        print("GLUE SCRIPT: Glue Job initialised successfully.")

        log_info(
            logger,
            "DEBUG: JOB START – Stable Training Dataset Update "
            f"DEBUG: JOB_NAME={JOB_NAME} DEBUG: INPUT_BUCKET={INPUT_BUCKET} "
            f"DEBUG: OUTPUT_BUCKET={OUTPUT_BUCKET} DEBUG: vendor_name={vendor_name} "
            f"DEBUG: prepared_input_key={prepared_input_key}",
        )
        print(
            "DEBUG: JOB START – Stable Training Dataset Update "
            f"DEBUG: JOB_NAME={JOB_NAME} DEBUG: INPUT_BUCKET={INPUT_BUCKET} "
            f"DEBUG: OUTPUT_BUCKET={OUTPUT_BUCKET} DEBUG: vendor_name={vendor_name}"
        )
        print(f"DEBUG: prepared_output_prefix={prepared_output_prefix}")

    except Exception as e:
        print(f"FATAL: Failed during {current_step}: {e}")
        raise

    s3_client = boto3.client("s3")

    # Thresholds (unchanged)
    MIN_PRODUCTS_FOR_TRAINING = 10
    MIN_KEYWORD_INSIDE_SUPPORT = 10
    MAX_KEYWORD_HARD_OUTSIDE = 0
    MIN_VENDOR_INSIDE_SUPPORT = 3
    MAX_VENDOR_OUTSIDE_SUPPORT = 0

    try:
        # =========================================================
        # STEP A: Locate input files
        # =========================================================
        current_step = "STEP A: Locate input files"

        one_vendor_match_key = (
            f"{prepared_output_prefix.rstrip('/')}/"
            f"{vendor_name}_categoryMatchingProposals_oneVendor_to_onePim_match.json"
        )

        log_info(
            logger,
            f"DEBUG: STEP A – expecting oneVendor_to_onePim_match at: "
            f"s3://{INPUT_BUCKET}/{one_vendor_match_key}",
        )
        print(
            "DEBUG: STEP A – expecting oneVendor_to_onePim_match at: "
            f"s3://{INPUT_BUCKET}/{one_vendor_match_key}"
        )

        if not s3_key_exists(s3_client, INPUT_BUCKET, one_vendor_match_key):
            msg = (
                "STEP A: oneVendor_to_onePim_match file not found at "
                f"s3://{INPUT_BUCKET}/{one_vendor_match_key}"
            )
            log_warning(logger, msg)
            print("[LOG-WARN-FAILED]", msg)
            raise RuntimeError(msg)

        log_info(logger, "DEBUG: STEP A – oneVendor_to_onePim_match file exists")
        print("DEBUG: STEP A – oneVendor_to_onePim_match file exists")

        stable_training_key = (
            "canonical_mappings/Category_Mapping_StableTrainingDataset.json"
        )
        training_subset_key = (
            "canonical_mappings/Category_Mapping_StableTrainingDataset_training.json"
        )

        log_info(
            logger,
            "DEBUG: STEP A – expecting Stable Training Dataset at: "
            f"s3://{INPUT_BUCKET}/{stable_training_key}",
        )
        print(
            "DEBUG: STEP A – expecting Stable Training Dataset at: "
            f"s3://{INPUT_BUCKET}/{stable_training_key}"
        )

        # =========================================================
        # STEP B1: Read oneVendor_to_onePim_match JSON
        # =========================================================
        current_step = "STEP B1: Read oneVendor_to_onePim_match JSON"

        log_info(
            logger,
            f"DEBUG: STEP B1 – reading oneVendor_to_onePim_match JSON from: "
            f"s3://{INPUT_BUCKET}/{one_vendor_match_key}",
        )
        print(
            "DEBUG: STEP B1 – reading oneVendor_to_onePim_match JSON from: "
            f"s3://{INPUT_BUCKET}/{one_vendor_match_key}"
        )

        match_data = read_json_from_s3(s3_client, INPUT_BUCKET, one_vendor_match_key)
        if not isinstance(match_data, dict):
            raise RuntimeError(
                "STEP B1: oneVendor_to_onePim_match JSON is not a dict at top-level."
            )

        vendor_categories = list(match_data.keys())
        log_info(
            logger,
            f"DEBUG: STEP B1 – vendor categories in match_data: {len(vendor_categories)}",
        )
        print(
            "DEBUG: STEP B1 – vendor categories in match_data:",
            len(vendor_categories),
        )

        # =========================================================
        # STEP B2: Extract new training records for assignment_source == 'existing_category_match'
        # =========================================================
        current_step = (
            "STEP B2: Extract training records with assignment_source == existing_category_match"
        )

        log_info(
            logger,
            "DEBUG: STEP B2 – extracting products where assignment_source == 'existing_category_match'",
        )
        print(
            "DEBUG: STEP B2 – extracting products where assignment_source == 'existing_category_match'"
        )

        new_training_records: List[Dict[str, Any]] = []

        for vcat_key, vcat_record in match_data.items():
            if not isinstance(vcat_record, dict):
                continue

            # Prefer explicit vendor_category_id field if present, otherwise fall back to the dict key.
            vcat_id = vcat_record.get("vendor_category_id") or vcat_key
            vcat_name = vcat_record.get("vendor_category_name")
            vcat_path = vcat_record.get("vendor_category_path")

            pim_matches = vcat_record.get("pim_matches") or []
            for pim_match in pim_matches:
                if not isinstance(pim_match, dict):
                    continue

                # IMPORTANT:
                # assignment_source is defined on pim_match-level in the original, working logic.
                # We stay compatible with that and only fall back to the vendor-category level
                # if the field is missing on the pim_match itself.
                assignment_source = pim_match.get("assignment_source")
                if assignment_source is None:
                    assignment_source = vcat_record.get("assignment_source")

                if assignment_source != "existing_category_match":
                    continue

                pim_cat_id = pim_match.get("pim_category_id")
                pim_cat_name = pim_match.get("pim_category_name")
                pim_cat_path = pim_match.get("pim_category_path")

                products = pim_match.get("products") or []
                for product in products:
                    if not isinstance(product, dict):
                        continue

                    article_id = product.get("article_id")
                    desc_short = product.get("description_short")
                    keywords = product.get("keywords")
                    class_codes = product.get("class_codes") or []

                    raw_rec = {
                        "pim_category_id": pim_cat_id,
                        "pim_category_name": pim_cat_name,
                        "pim_category_path": pim_cat_path,
                        "vendor_short_name": vendor_name,
                        "vendor_category_id": vcat_id,
                        "vendor_category_name": vcat_name,
                        "vendor_category_path": vcat_path,
                        "article_id": article_id,
                        "description_short": desc_short,
                        "keywords": keywords,
                        "class_codes": class_codes,
                    }

                    norm_rec = normalise_training_record(raw_rec)
                    new_training_records.append(norm_rec)

        log_info(
            logger,
            f"DEBUG: STEP B2 – new_training_records count: {len(new_training_records)}",
        )
        print(
            "DEBUG: STEP B2 – new_training_records count:",
            len(new_training_records),
        )

        if not new_training_records:
            log_warning(
                logger,
                "DEBUG: STEP B2 – no new_training_records; will reuse existing Stable "
                "Training Dataset only",
            )
            print(
                "DEBUG: STEP B2 – no new_training_records; will reuse existing Stable "
                "Training Dataset only"
            )

        # =========================================================
        # STEP B3: Merge with existing Stable Training Dataset
        # =========================================================
        current_step = "STEP B3: Merge new training records into Stable Training Dataset"

        log_info(logger, "DEBUG: STEP B3 – merging new training records into Stable Training Dataset")
        print("DEBUG: STEP B3 – merging new training records into Stable Training Dataset")

        if s3_key_exists(s3_client, INPUT_BUCKET, stable_training_key):
            existing_data = read_json_from_s3(
                s3_client, INPUT_BUCKET, stable_training_key
            )
            if isinstance(existing_data, list):
                existing_records = [
                    normalise_training_record(rec) for rec in existing_data
                ]
            else:
                existing_records = []
        else:
            existing_records = []

        log_info(
            logger,
            f"DEBUG: STEP B3 – existing_records (before merge): {len(existing_records)}",
        )
        print(
            "DEBUG: STEP B3 – existing_records (before merge):",
            len(existing_records),
        )

        # Merge with de-duplication by (vendor_short_name, article_id).
        # New training records always overwrite older ones for the same key,
        # so the Stable Training Dataset keeps the latest canonical mapping
        # per vendor article while avoiding duplicates.
        combined_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for rec in existing_records:
            v_short = rec.get("vendor_short_name") or ""
            art_id = rec.get("article_id") or ""
            combined_by_key[(v_short, art_id)] = rec

        for rec in new_training_records:
            v_short = rec.get("vendor_short_name") or ""
            art_id = rec.get("article_id") or ""
            combined_by_key[(v_short, art_id)] = rec

        final_records = list(combined_by_key.values())
        dedup_removed = (len(existing_records) + len(new_training_records)) - len(final_records)

        log_info(
            logger,
            f"DEBUG: STEP B3 – final_records (after merge & de-dup): {len(final_records)} "
            f"(duplicates removed={dedup_removed})",
        )
        print(
            "DEBUG: STEP B3 – final_records (after merge & de-dup):",
            len(final_records),
            "(duplicates removed=",
            dedup_removed,
            ")",
        )

        # =========================================================
        # STEP B4: Write Stable Training Dataset back to S3
        # =========================================================
        current_step = "STEP B4: Write Stable Training Dataset to S3"

        log_info(
            logger,
            f"DEBUG: STEP B4 – writing Stable Training Dataset to s3://{INPUT_BUCKET}/{stable_training_key}",
        )
        print(
            "DEBUG: STEP B4 – writing Stable Training Dataset to s3://"
            f"{INPUT_BUCKET}/{stable_training_key}"
        )

        write_json_to_s3(
            s3_client,
            INPUT_BUCKET,
            stable_training_key,
            final_records,
            indent=2,
            logger=logger,
        )

        log_info(logger, "DEBUG: STEP B4 – Stable Training Dataset write completed")
        print("DEBUG: STEP B4 – Stable Training Dataset write completed")

        # =========================================================
        # STEP C: Build in-memory training subset (and write mirror JSON)
        # =========================================================
        current_step = (
            "STEP C: Build in-memory training subset and mirror JSON file"
        )

        log_info(
            logger,
            "DEBUG: STEP C – deriving training subset with "
            f"MIN_PRODUCTS_FOR_TRAINING={MIN_PRODUCTS_FOR_TRAINING}",
        )
        print(
            "DEBUG: STEP C – deriving training subset with "
            f"MIN_PRODUCTS_FOR_TRAINING={MIN_PRODUCTS_FOR_TRAINING}"
        )

        training_records = []

        # Count products per pim_category_id in final_records
        category_counts: Dict[str, int] = {}
        for rec in final_records:
            cid = rec.get("pim_category_id")
            if cid is None:
                continue
            cid_str = str(cid)
            category_counts[cid_str] = category_counts.get(cid_str, 0) + 1

        usable_categories = [
            cid_str
            for cid_str, count in category_counts.items()
            if count >= MIN_PRODUCTS_FOR_TRAINING
        ]

        log_info(
            logger,
            "DEBUG: STEP C – distinct pim_category_id count: "
            f"{len(category_counts)}",
        )
        print(
            "DEBUG: STEP C – distinct pim_category_id count:",
            len(category_counts),
        )
        log_info(
            logger,
            "DEBUG: STEP C – usable_categories (>= "
            f"{MIN_PRODUCTS_FOR_TRAINING} products): "
            f"{len(usable_categories)} -> {usable_categories}",
        )
        print(
            "DEBUG: STEP C – usable_categories (>= "
            f"{MIN_PRODUCTS_FOR_TRAINING} products):",
            len(usable_categories),
            "->",
            usable_categories,
        )

        if usable_categories:
            for rec in final_records:
                cid = rec.get("pim_category_id")
                if cid is None:
                    continue
                cid_str = str(cid)
                if cid_str in usable_categories:
                    training_records.append(rec)

        log_info(
            logger,
            f"DEBUG: STEP C – training_records count after filtering: {len(training_records)}",
        )
        print(
            "DEBUG: STEP C – training_records count after filtering:",
            len(training_records),
        )

        previous_training_records = []

        if s3_key_exists(s3_client, INPUT_BUCKET, training_subset_key):
            prev_data = read_json_from_s3(
                s3_client, INPUT_BUCKET, training_subset_key
            )
            if isinstance(prev_data, list):
                previous_training_records = [
                    normalise_training_record(rec) for rec in prev_data
                ]
            else:
                previous_training_records = []
        else:
            previous_training_records = []

        prev_category_counts: Dict[str, int] = {}
        for rec in previous_training_records:
            cid = rec.get("pim_category_id")
            if cid is None:
                continue
            cid_str = str(cid)
            prev_category_counts[cid_str] = prev_category_counts.get(cid_str, 0) + 1

        prev_usable_categories = [
            cid_str
            for cid_str, count in prev_category_counts.items()
            if count >= MIN_PRODUCTS_FOR_TRAINING
        ]

        log_info(
            logger,
            "DEBUG: STEP C – previous training distinct pim_category_id count: "
            f"{len(prev_category_counts)}",
        )
        print(
            "DEBUG: STEP C – previous training distinct pim_category_id count:",
            len(prev_category_counts),
        )
        log_info(
            logger,
            "DEBUG: STEP C – previous usable_categories (>= "
            f"{MIN_PRODUCTS_FOR_TRAINING} products): "
            f"{len(prev_usable_categories)} -> {prev_usable_categories}",
        )
        print(
            "DEBUG: STEP C – previous usable_categories (>= "
            f"{MIN_PRODUCTS_FOR_TRAINING} products):",
            len(prev_usable_categories),
            "->",
            prev_usable_categories,
        )

        log_info(
            logger,
            f"DEBUG: STEP C – previous_training_records count after filtering: {len(previous_training_records)}",
        )
        print(
            "DEBUG: STEP C – previous_training_records count after filtering:",
            len(previous_training_records),
        )

        if training_records:
            log_info(
                logger,
                "DEBUG: STEP C – sample training_record type: "
                f"{type(training_records[0]).__name__}",
            )
            print(
                "DEBUG: STEP C – sample training_record type:",
                type(training_records[0]).__name__,
            )
            log_info(
                logger,
                "DEBUG: STEP C – sample training_record keys: "
                f"{list(training_records[0].keys())}",
            )
            print(
                "DEBUG: STEP C – sample training_record keys:",
                list(training_records[0].keys()),
            )

        write_json_to_s3(
            s3_client,
            INPUT_BUCKET,
            training_subset_key,
            training_records,
            indent=2,
            logger=logger,
        )

        # =========================================================
        # STEP D: Build KEYWORD statistics for K1 and K2
        # =========================================================
        current_step = "STEP D: Build KEYWORD statistics and rule proposals"

        log_info(
            logger,
            f"DEBUG: STEP D – starting rule proposal generation; training_records={len(training_records)}",
        )
        print(
            "DEBUG: STEP D – starting rule proposal generation; "
            f"training_records={len(training_records)}"
        )

        # ---------------------------------------------------------
        # STEP D1: Create Spark DataFrame from training_records
        # ---------------------------------------------------------
        current_step = "STEP D1: Create Spark DataFrame from training_records"
        log_info(logger, "DEBUG: STEP D1 – creating DataFrame from training_records")
        print("DEBUG: STEP D1 – creating DataFrame from training_records")

        training_schema = T.StructType(
            [
                T.StructField("pim_category_id", T.StringType(), True),
                T.StructField("pim_category_name", T.StringType(), True),
                T.StructField("pim_category_path", T.StringType(), True),
                T.StructField("vendor_short_name", T.StringType(), True),
                T.StructField("vendor_category_id", T.StringType(), True),
                T.StructField("vendor_category_name", T.StringType(), True),
                T.StructField("vendor_category_path", T.StringType(), True),
                T.StructField("article_id", T.StringType(), True),
                T.StructField("description_short", T.StringType(), True),
                T.StructField("keywords", T.ArrayType(T.StringType()), True),
                T.StructField("class_codes", T.ArrayType(T.StringType()), True),
            ]
        )

        if not training_records:
            log_warning(
                logger,
                "STEP D1: No training_records available after filtering; "
                "cannot compute keyword statistics.",
            )
            print(
                "[LOG-WARN-FAILED] STEP D1: No training_records available; "
                "cannot compute keyword statistics."
            )
            training_df = spark.createDataFrame([], schema=training_schema)
        else:
            training_df = spark.createDataFrame(training_records, schema=training_schema)

        log_info(
            logger,
            f"DEBUG: STEP D1 – training_records length: {len(training_records)}",
        )
        print(
            "DEBUG: STEP D1 – training_records length:",
            len(training_records),
        )

        if training_records:
            sample_rec = training_records[0]
            print("DEBUG: STEP D1 – sample record type:")
            print(type(sample_rec).__name__)
            print("DEBUG: STEP D1 – sample record keys:")
            print(list(sample_rec.keys()))
            print("DEBUG: STEP D1 – sample record:")
            print(
                json.dumps(
                    sample_rec, indent=2, ensure_ascii=False
                )[:1000]
            )

        # ---------------------------------------------------------
        # STEP D1 (previous training): Create DataFrame for previous_training_records
        # ---------------------------------------------------------
        prev_training_df = spark.createDataFrame(
            previous_training_records, schema=training_schema
        )

        prev_kw_df = (
            prev_training_df
            .withColumn("keyword", F.explode(F.col("keywords")))
            .select("pim_category_id", "keyword")
            .distinct()
        )

        # ---------------------------------------------------------
        # STEP D2: Compute KEYWORD statistics at training level
        # ---------------------------------------------------------
        current_step = "STEP D2: Compute KEYWORD statistics at training level"
        log_info(
            logger, "DEBUG: STEP D2 – exploding KEYWORD terms from training subset"
        )
        print("DEBUG: STEP D2 – exploding KEYWORD terms from training subset")

        kw_train_df = (
            training_df
            .withColumn("keyword", F.explode(F.col("keywords")))
            .select(
                "pim_category_id",
                "article_id",
                "keyword",
            )
        )

        log_info(
            logger,
            "DEBUG: STEP D2 – computing inside_count and train_total_count",
        )
        print("DEBUG: STEP D2 – computing inside_count and train_total_count")

        train_insides = (
            kw_train_df.groupBy("pim_category_id", "keyword")
            .agg(
                F.countDistinct("article_id").alias("inside_count"),
            )
        )

        keyword_global = (
            kw_train_df.groupBy("keyword")
            .agg(
                F.countDistinct("article_id").alias("train_total_count"),
            )
        )

        df_stats = (
            train_insides.join(keyword_global, on="keyword", how="left")
            .withColumn(
                "hard_outside_count",
                F.col("train_total_count") - F.col("inside_count"),
            )
        )

        log_info(
            logger,
            "DEBUG: STEP D2 – completed inside_count/train_total_count/hard_outside_count computation",
        )
        print(
            "DEBUG: STEP D2 – completed inside_count/train_total_count/hard_outside_count computation"
        )

        # ---------------------------------------------------------
        # STEP D3: Compute vendor KEYWORD statistics for the current vendor
        # ---------------------------------------------------------
        current_step = "STEP D3: Compute vendor KEYWORD statistics"
        log_info(
            logger, "DEBUG: STEP D3 – building vendor KEYWORD statistics from match_data"
        )
        print("DEBUG: STEP D3 – building vendor KEYWORD statistics from match_data")

        vendor_kw_rows: List[Dict[str, Any]] = []

        for vcat_key, vcat_record in match_data.items():
            if not isinstance(vcat_record, dict):
                continue

            vcat_id = vcat_record.get("vendor_category_id")
            pim_matches = vcat_record.get("pim_matches") or []
            for pim_match in pim_matches:
                if not isinstance(pim_match, dict):
                    continue

                pim_cat_id = pim_match.get("pim_category_id")
                products = pim_match.get("products") or []
                for product in products:
                    if not isinstance(product, dict):
                        continue

                    article_id = product.get("article_id")
                    raw_keywords = product.get("keywords")
                    keywords = normalise_keywords(raw_keywords)

                    for kw in keywords:
                        vendor_kw_rows.append(
                            {
                                "pim_category_id": str(pim_cat_id),
                                "article_id": str(article_id),
                                "keyword": kw,
                            }
                        )

        vendor_kw_schema = T.StructType(
            [
                T.StructField("pim_category_id", T.StringType(), True),
                T.StructField("article_id", T.StringType(), True),
                T.StructField("keyword", T.StringType(), True),
            ]
        )

        if vendor_kw_rows:
            vendor_kw_df = spark.createDataFrame(vendor_kw_rows, schema=vendor_kw_schema)
        else:
            vendor_kw_df = spark.createDataFrame([], schema=vendor_kw_schema)

        log_info(
            logger,
            "DEBUG: STEP D3 – built vendor_inside_count/vendor_total_count DataFrames",
        )
        print(
            "DEBUG: STEP D3 – built vendor_inside_count/vendor_total_count DataFrames"
        )

        vendor_inside_df = (
            vendor_kw_df.groupBy("pim_category_id", "keyword")
            .agg(
                F.countDistinct("article_id").alias("vendor_inside_count"),
            )
        )

        vendor_global_df = (
            vendor_kw_df.groupBy("keyword")
            .agg(
                F.countDistinct("article_id").alias("vendor_total_count"),
            )
        )

        df_stats = (
            df_stats.join(
                vendor_inside_df,
                on=["pim_category_id", "keyword"],
                how="left",
            )
            .join(
                vendor_global_df,
                on="keyword",
                how="left",
            )
        )

        df_stats = df_stats.fillna(
            {
                "vendor_inside_count": 0,
                "vendor_total_count": 0,
            }
        )

        df_stats = df_stats.withColumn(
            "vendor_outside_count",
            F.col("vendor_total_count") - F.col("vendor_inside_count"),
        )

        log_info(
            logger,
            "DEBUG: STEP D3 – joined training and vendor statistics and vendor_outside_count computation",
        )
        print(
            "DEBUG: STEP D3 – joined training and vendor statistics and vendor_outside_count computation"
        )

        # ---------------------------------------------------------
        # STEP D5: Apply thresholds (K1) and build rule_status + K2 extension
        # ---------------------------------------------------------
        current_step = "STEP D5: Apply thresholds and derive rule status"
        log_info(
            logger,
            "DEBUG: STEP D5 – applying thresholds and deriving rule status: "
            f"MIN_KEYWORD_INSIDE_SUPPORT={MIN_KEYWORD_INSIDE_SUPPORT}, "
            f"MAX_KEYWORD_HARD_OUTSIDE={MAX_KEYWORD_HARD_OUTSIDE}, "
            f"MIN_VENDOR_INSIDE_SUPPORT={MIN_VENDOR_INSIDE_SUPPORT}, "
            f"MAX_VENDOR_OUTSIDE_SUPPORT={MAX_VENDOR_OUTSIDE_SUPPORT}",
        )
        print(
            "DEBUG: STEP D5 – applying thresholds and deriving rule status: "
            f"MIN_KEYWORD_INSIDE_SUPPORT={MIN_KEYWORD_INSIDE_SUPPORT}, "
            f"MAX_KEYWORD_HARD_OUTSIDE={MAX_KEYWORD_HARD_OUTSIDE}, "
            f"MIN_VENDOR_INSIDE_SUPPORT={MIN_VENDOR_INSIDE_SUPPORT}, "
            f"MAX_VENDOR_OUTSIDE_SUPPORT={MAX_VENDOR_OUTSIDE_SUPPORT}"
        )

        df_candidates = df_stats.where(
            (F.col("inside_count") >= F.lit(MIN_KEYWORD_INSIDE_SUPPORT))
            & (F.col("hard_outside_count") <= F.lit(MAX_KEYWORD_HARD_OUTSIDE))
            & (F.col("vendor_inside_count") >= F.lit(MIN_VENDOR_INSIDE_SUPPORT))
            & (F.col("vendor_outside_count") <= F.lit(MAX_VENDOR_OUTSIDE_SUPPORT))
        )

        conservative_count = df_candidates.limit(1).count()
        if conservative_count > 0:
            log_info(
                logger,
                "DEBUG: STEP D5 – at least one candidate KEYWORD term passes conservative filters",
            )
            print(
                "DEBUG: STEP D5 – at least one candidate KEYWORD term passes conservative filters"
            )
        else:
            log_info(
                logger,
                "DEBUG: STEP D5 – no KEYWORD term passes conservative filters; K1 rules will be empty",
            )
            print(
                "DEBUG: STEP D5 – no KEYWORD term passes conservative filters; K1 rules will be empty"
            )

        current_step = "STEP D5: Join with previous training pairs for new-rule detection"
        log_info(
            logger,
            "DEBUG: STEP D5 – building previous training KEYWORD pairs from previous_training_records",
        )
        print(
            "DEBUG: STEP D5 – building previous training KEYWORD pairs from previous_training_records",
        )

        try:
            df_prev_kw = prev_kw_df
            prev_kw_distinct = df_prev_kw.count()
            log_info(
                logger,
                f"DEBUG: STEP D5 – distinct (pim_category_id, keyword) pairs in previous training: {prev_kw_distinct}",
            )
            print(
                "DEBUG: STEP D5 – distinct (pim_category_id, keyword) pairs in previous training:",
                prev_kw_distinct,
            )
        except Exception as e:
            log_warning(
                logger,
                f"STEP D5: Could not create DataFrame from previous_training_records. "
                f"K1 new rule detection will treat all candidates as 'new'. | {e}",
            )
            print(
                "[LOG-WARN-FAILED] STEP D5: Could not create DataFrame from previous_training_records. "
                "K1 new rule detection will treat all candidates as 'new'."
            )
            df_prev_kw = spark.createDataFrame(
                [],
                schema=T.StructType(
                    [
                        T.StructField("pim_category_id", T.StringType(), True),
                        T.StructField("keyword", T.StringType(), True),
                    ]
                ),
            )

        df_stats = (
            df_stats.join(
                df_prev_kw.withColumn("known_in_previous_training", F.lit(True)),
                on=["pim_category_id", "keyword"],
                how="left",
            )
            .fillna({"known_in_previous_training": False})
        )

        df_stats = (
            df_stats.withColumn(
                "has_vendor_usage",
                F.col("vendor_total_count") > F.lit(0),
            )
            .withColumn(
                "global_candidate",
                (F.col("inside_count") >= F.lit(MIN_KEYWORD_INSIDE_SUPPORT))
                & (F.col("hard_outside_count") <= F.lit(MAX_KEYWORD_HARD_OUTSIDE)),
            )
            .withColumn(
                "vendor_supports",
                F.col("global_candidate")
                & (F.col("vendor_inside_count") >= F.lit(MIN_VENDOR_INSIDE_SUPPORT))
                & (F.col("vendor_outside_count") <= F.lit(MAX_VENDOR_OUTSIDE_SUPPORT)),
            )
            .withColumn(
                "vendor_violates",
                F.col("global_candidate")
                & (F.col("vendor_outside_count") > F.lit(MAX_VENDOR_OUTSIDE_SUPPORT)),
            )
            .withColumn(
                "is_new_rule",
                F.col("global_candidate") & (~F.col("known_in_previous_training")),
            )
        )

        current_step = "STEP D5: Classify vendor_status per rule"
        log_info(logger, "DEBUG: STEP D5 – classifying vendor_status per (category, keyword)")
        print("DEBUG: STEP D5 – classifying vendor_status per (category, keyword)")

        df_stats = df_stats.withColumn(
            "vendor_status",
            F.when(
                F.col("is_new_rule") & F.col("vendor_supports"),
                F.lit("new"),
            )
            .when(
                F.col("global_candidate") & F.col("vendor_supports"),
                F.lit("supported"),
            )
            .when(
                F.col("global_candidate") & F.col("vendor_violates"),
                F.lit("violated"),
            )
            .when(
                F.col("global_candidate") & (~F.col("has_vendor_usage")),
                F.lit("not_impacted"),
            )
            .otherwise(F.lit(None)),
        )

        # Build mapping from pim_category_id -> pim_category_name/path from training_records
        category_name_map = {}
        category_path_map = {}
        for rec in training_records:
            cid = rec.get("pim_category_id")
            if cid is None:
                continue
            cid_str = str(cid)
            if cid_str not in category_name_map and rec.get("pim_category_name") is not None:
                category_name_map[cid_str] = rec.get("pim_category_name")
            if cid_str not in category_path_map and rec.get("pim_category_path") is not None:
                category_path_map[cid_str] = rec.get("pim_category_path")

        # ---------------------------------------------------------------------
        # Build helper indices for K2 logic (contains_any_exclude_any rules)
        # We reuse the in-memory training_records and match_data so we do not
        # change the existing Spark-based counting logic for K1.
        # ---------------------------------------------------------------------
        print("DEBUG: STEP D5 – building helper indices for K2 logic")
        training_kw_occurrences = {}
        for rec in training_records:
            cid = rec.get("pim_category_id")
            if cid is None:
                continue
            cid_str = str(cid)
            kw_list = rec.get("keywords") or []
            # keywords in training_records are already normalised lists of strings
            kw_set = set([k for k in kw_list if isinstance(k, str) and k.strip() != ""])
            if not kw_set:
                continue
            for kw in kw_set:
                training_kw_occurrences.setdefault(kw, []).append(
                    {
                        "pim_category_id": cid_str,
                        "keywords_set": kw_set,
                    }
                )

        print(
            "DEBUG: STEP D5 – training_kw_occurrences built for keywords:",
            len(training_kw_occurrences),
        )

        # Build vendor keyword occurrences per keyword from match_data
        vendor_kw_occurrences = {}
        try:
            for vcat_key, vcat_record in match_data.items():
                if not isinstance(vcat_record, dict):
                    continue
                pim_matches = vcat_record.get("pim_matches") or []
                for pim_match in pim_matches:
                    if not isinstance(pim_match, dict):
                        continue
                    pim_category_id = pim_match.get("pim_category_id")
                    pim_category_id_str = (
                        str(pim_category_id) if pim_category_id is not None else None
                    )
                    products = pim_match.get("products") or []
                    for product in products:
                        if not isinstance(product, dict):
                            continue
                        raw_keywords = product.get("keywords")
                        if raw_keywords is None:
                            kw_list = []
                        elif isinstance(raw_keywords, list):
                            kw_list = [
                                str(k)
                                for k in raw_keywords
                                if k is not None and str(k).strip() != ""
                            ]
                        else:
                            kw_str = str(raw_keywords)
                            kw_list = [kw_str] if kw_str.strip() != "" else []
                        if not kw_list:
                            continue
                        kw_set = set(kw_list)
                        for kw in kw_set:
                            kw_s = kw.strip()
                            if not kw_s:
                                continue
                            vendor_kw_occurrences.setdefault(kw_s, []).append(
                                {
                                    "pim_category_id": pim_category_id_str,
                                    "keywords_set": kw_set,
                                }
                            )
        except Exception as e:
            log_warning(
                logger,
                f"STEP D5: Failed to build vendor_kw_occurrences for K2 logic: {e}",
            )

        print(
            "DEBUG: STEP D5 – vendor_kw_occurrences built for keywords:",
            len(vendor_kw_occurrences),
        )

        # ---------------------------------------------------------------------
        # K1 RULES (unchanged): contains_any with hard_outside_count == 0
        # ---------------------------------------------------------------------
        # Keep only rules that actually meet the global candidate thresholds
        # and have a meaningful vendor_status (new / supported / violated / not_impacted).
        df_output_k1 = df_stats.where(
            F.col("global_candidate") & F.col("vendor_status").isNotNull()
        )

        print(
            "DEBUG: STEP D5 – df_output_k1.count() will be collected on driver"
        )
        rows_k1 = df_output_k1.collect()
        rule_proposals = []

        for r in rows_k1:
            pim_category_id = r["pim_category_id"]
            pim_category_id_str = (
                str(pim_category_id) if pim_category_id is not None else None
            )
            pim_category_name = category_name_map.get(pim_category_id_str)
            keyword = r["keyword"]

            inside_count = int(r["inside_count"]) if r["inside_count"] is not None else 0
            train_total_count = (
                int(r["train_total_count"]) if r["train_total_count"] is not None else 0
            )
            hard_outside_count = (
                int(r["hard_outside_count"]) if r["hard_outside_count"] is not None else 0
            )
            vendor_inside_count = (
                int(r["vendor_inside_count"]) if r["vendor_inside_count"] is not None else 0
            )
            vendor_total_count = (
                int(r["vendor_total_count"]) if r["vendor_total_count"] is not None else 0
            )
            vendor_outside_count = (
                int(r["vendor_outside_count"]) if r["vendor_outside_count"] is not None else 0
            )
            vendor_status = r["vendor_status"]

            rule_obj = {
                "pim_category_id": pim_category_id,
                "pim_category_name": pim_category_name,
                "field_name": "KEYWORD",
                "operator": "contains_any",
                "values_include": [keyword],
                "values_exclude": [],
                "stats": {
                    "inside_count": inside_count,
                    "train_total_count": train_total_count,
                    "hard_outside_count": hard_outside_count,
                    "vendor_inside_count": vendor_inside_count,
                    "vendor_total_count": vendor_total_count,
                    "vendor_outside_count": vendor_outside_count,
                    "vendor_status": vendor_status,
                },
            }
            rule_proposals.append(rule_obj)

        print(
            "DEBUG: STEP D5 – K1 rule proposals built:", len(rule_proposals)
        )

        # ---------------------------------------------------------------------
        # K2 RULES: contains_any_exclude_any for keywords where K1 fails only
        # because hard_outside_count > 0.
        # ---------------------------------------------------------------------
        print("DEBUG: STEP D5 – starting K2 candidate evaluation")
        df_k2_candidates = df_stats.where(
            (F.col("inside_count") >= F.lit(MIN_KEYWORD_INSIDE_SUPPORT))
            & (F.col("hard_outside_count") > F.lit(MAX_KEYWORD_HARD_OUTSIDE))
        )

        k2_rows = df_k2_candidates.select(
            "pim_category_id",
            "keyword",
            "inside_count",
            "train_total_count",
            "hard_outside_count",
            "known_in_previous_training",
        ).collect()

        print(
            "DEBUG: STEP D5 – K2 candidate rows collected:",
            len(k2_rows),
        )

        accepted_k2_rules = 0

        for r in k2_rows:
            pim_category_id = r["pim_category_id"]
            if pim_category_id is None:
                continue
            pim_category_id_str = str(pim_category_id)
            keyword = r["keyword"]
            if not keyword:
                continue
            keyword_str = str(keyword)

            # Occurrences of this keyword in training (inside vs outside the category)
            occ_list = training_kw_occurrences.get(keyword_str)
            if not occ_list:
                continue

            inside_sets = [
                o["keywords_set"]
                for o in occ_list
                if o["pim_category_id"] == pim_category_id_str
            ]
            outside_sets = [
                o["keywords_set"]
                for o in occ_list
                if o["pim_category_id"] != pim_category_id_str
            ]

            # By definition of K2 we only care where there are outside uses
            if not outside_sets:
                continue

            inside_tokens = set()
            for s in inside_sets:
                inside_tokens |= s

            outside_tokens = set()
            for s in outside_sets:
                outside_tokens |= s

            # Exclude candidates: tokens that appear only in outside sets,
            # never in the inside sets for this (category, keyword).
            exclude_candidates = outside_tokens - inside_tokens
            if not exclude_candidates:
                # There is no token that is unique to outside categories,
                # so we cannot cleanly eliminate outside usage via excludes.
                continue

            # Coverage check: every outside occurrence of the keyword must contain
            # at least one exclude token, otherwise hard_outside_count after
            # applying K2 would still be > 0.
            coverage_ok = True
            for s in outside_sets:
                if not (s & exclude_candidates):
                    coverage_ok = False
                    break

            if not coverage_ok:
                # We cannot achieve hard_outside_count == 0 without excluding
                # tokens that are also present in the inside category, so skip.
                continue

            # At this point K2 can be constructed with E = exclude_candidates.
            # By construction:
            # - inside_count_K2 == inside_count_K1
            # - hard_outside_count_K2 == 0
            inside_count = int(r["inside_count"]) if r["inside_count"] is not None else 0
            if inside_count < MIN_KEYWORD_INSIDE_SUPPORT:
                # Safety check; should not happen due to df_k2_candidates filter.
                continue

            # Compute vendor statistics under K2 semantics:
            vendor_occ_list = vendor_kw_occurrences.get(keyword_str, [])
            vendor_inside_k2 = 0
            vendor_total_k2 = 0
            vendor_outside_k2 = 0

            for occ in vendor_occ_list:
                v_cid = occ.get("pim_category_id")
                kw_set = occ.get("keywords_set") or set()
                # Skip contexts that contain any exclude token:
                if kw_set & exclude_candidates:
                    continue
                vendor_total_k2 += 1
                if v_cid == pim_category_id_str:
                    vendor_inside_k2 += 1
                else:
                    vendor_outside_k2 += 1

            # Derive vendor_status for K2
            if vendor_total_k2 == 0:
                vendor_status_k2 = "not_impacted"
            elif (
                vendor_inside_k2 >= MIN_VENDOR_INSIDE_SUPPORT
                and vendor_outside_k2 <= MAX_VENDOR_OUTSIDE_SUPPORT
            ):
                # Use previous-training information to distinguish "new" vs "supported"
                known_prev = bool(r["known_in_previous_training"])
                vendor_status_k2 = "supported" if known_prev else "new"
            else:
                # Vendor violates the K2 rule; do not propose it.
                continue

            # K2 global candidate is always "clean" from training perspective:
            # we force hard_outside_count_K2 to be 0 while keeping inside_count
            # identical to K1.
            values_exclude = sorted(exclude_candidates)

            rule_obj_k2 = {
                "pim_category_id": pim_category_id,
                "pim_category_name": category_name_map.get(pim_category_id_str),
                "field_name": "KEYWORD",
                "operator": "contains_any_exclude_any",
                "values_include": [keyword_str],
                "values_exclude": values_exclude,
                "stats": {
                    "inside_count": inside_count,
                    # After applying excludes all remaining uses are inside the category.
                    "train_total_count": inside_count,
                    "hard_outside_count": 0,
                    "vendor_inside_count": vendor_inside_k2,
                    "vendor_total_count": vendor_total_k2,
                    "vendor_outside_count": vendor_outside_k2,
                    "vendor_status": vendor_status_k2,
                },
            }

            rule_proposals.append(rule_obj_k2)
            accepted_k2_rules += 1

        print(
            "DEBUG: STEP D5 – accepted K2 rule proposals:",
            accepted_k2_rules,
        )

        log_info(
            logger,
            f"STEP D5: Generated {len(rule_proposals)} total rule-status entries "
            f"(K1 + K2) for vendor={vendor_name}.",
        )
        print(
            "DEBUG: STEP D5 – generated total rule-status entries: "
            f"{len(rule_proposals)}"
        )

        # =========================================================
        # STEP D6: Write rule proposals JSON to S3
        # =========================================================
        current_step = "STEP D6: Write rule proposals JSON to S3"

        rule_proposals_key = (
            f"{prepared_output_prefix.rstrip('/')}/"
            f"ruleProposals_{vendor_name}.json"
        )

        log_info(
            logger,
            f"DEBUG: STEP D6 – writing rule proposals to s3://{OUTPUT_BUCKET}/{rule_proposals_key}",
        )
        print(
            "DEBUG: STEP D6 – writing rule proposals to s3://"
            f"{OUTPUT_BUCKET}/{rule_proposals_key}"
        )

        write_json_to_s3(
            s3_client,
            OUTPUT_BUCKET,
            rule_proposals_key,
            rule_proposals,
            indent=2,
            logger=logger,
        )

        # =========================================================
        # STEP E: Update Category_Mapping_Reference with mapping_methods
        # =========================================================
        current_step = (
            "STEP E: Update Category_Mapping_Reference mapping_methods "
            "from rule_proposals"
        )

        latest_ref_key, reference_data = load_latest_category_mapping_reference(
            s3_client, INPUT_BUCKET, logger=logger
        )

        reference_index = build_reference_index(reference_data)
        updated_reference_list, change_log = update_mapping_methods_from_rule_proposals(
            logger,
            reference_index,
            rule_proposals,
        )

        timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        new_reference_key = (
            f"canonical_mappings/Category_Mapping_Reference_{timestamp_str}.json"
        )
        log_info(
            logger,
            "STEP E2: Writing updated Category_Mapping_Reference to "
            f"s3://{INPUT_BUCKET}/{new_reference_key}",
        )
        print(
            "STEP E2: Writing updated Category_Mapping_Reference to "
            f"s3://{INPUT_BUCKET}/{new_reference_key}"
        )

        write_json_to_s3(
            s3_client,
            INPUT_BUCKET,
            new_reference_key,
            updated_reference_list,
            indent=2,
            logger=logger,
        )

        change_log_key = (
            f"canonical_mappings/Category_Mapping_Reference_changelog_{timestamp_str}.json"
        )
        log_info(
            logger,
            "STEP E3: Writing Category_Mapping_Reference change log to "
            f"s3://{INPUT_BUCKET}/{change_log_key}",
        )
        print(
            "STEP E3: Writing Category_Mapping_Reference change log to "
            f"s3://{INPUT_BUCKET}/{change_log_key}"
        )

        write_json_to_s3(
            s3_client,
            INPUT_BUCKET,
            change_log_key,
            change_log,
            indent=2,
            logger=logger,
        )

        log_info(logger, "JOB COMPLETED SUCCESSFULLY")
        print("JOB COMPLETED SUCCESSFULLY")

        job.commit()

    except Exception as e:
        log_error(
            logger,
            f"FATAL: Job failed with exception in step: {current_step}\n{repr(e)}",
        )
        print("FATAL: Job failed with exception in step:")
        print(current_step)
        print(repr(e))
        raise


if __name__ == "__main__":
    main()
