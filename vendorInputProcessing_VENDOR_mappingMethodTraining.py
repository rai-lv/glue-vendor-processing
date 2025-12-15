#!/usr/bin/env python3
# Glue 5.0 / Spark / Python 3 script
#
# STEP: Update Stable Training Dataset per vendor.
#
# Triggered from Lambda/Make with payload:
# {
#   "job_name": "vendorInputProcessing_VENDOR_mappingMethodTraining",
#   "arguments": {
#     "vendor_name": "<VENDOR_NAME>",
#     "prepared_input_key": "location on S3",
#     "prepared_output_prefix": "location on S3"
#   }
# }
#
# Glue job has parameters: --JOB_NAME, --INPUT_BUCKET, --OUTPUT_BUCKET
#
# This script:
# - Reads s3://INPUT_BUCKET/<prepared_input_key>/canonicalCategoryMapping/
#       <vendor_name>_categoryMatchingProposals_oneVendor_to_onePim_match.json
# - Extracts products where assignment_source == "existing_category_match"
# - Appends/merges them into:
#       s3://INPUT_BUCKET/canonical_mappings/Category_Mapping_StableTrainingDataset.json
#   using (vendor_short_name, article_id) as dedup key.
# - Derives a training subset with only PIM categories that have at least
#   MIN_PRODUCTS_FOR_TRAINING products:
#       s3://INPUT_BUCKET/canonical_mappings/Category_Mapping_StableTrainingDataset_training.json
# - STEP D: From the training subset + this vendor's oneVendor_to_onePim file,
#   derive KEYWORD-based rule proposals per PIM category and
#   write them to:
#       s3://INPUT_BUCKET/canonical_mappings/
#           Category_Mapping_RuleProposals_<vendor_name>_<timestamp>.json
# - STEP E: Update Category_Mapping_Reference_<timestamp>.json based on rule
#   proposals (direct update, no manual rules) and write a change-log file:
#       s3://INPUT_BUCKET/canonical_mappings/
#           Category_Mapping_RuleChanges_<vendor_name>_<timestamp>.json

import sys
import json
import re
import traceback
import os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Signal as early as possible that the module was loaded
print("GLUE SCRIPT (mappingMethodTraining): module import started")


def log_info(logger, msg: str):
    """Safe wrapper around logger.info that falls back to print if logging fails."""
    try:
        logger.info(msg)
    except Exception as e:
        print(f"[LOG-INFO-FAILED] {msg} | {repr(e)}")


def log_warning(logger, msg: str):
    """Safe wrapper around logger.warning that falls back to print if logging fails."""
    try:
        logger.warning(msg)
    except Exception as e:
        print(f"[LOG-WARN-FAILED] {msg} | {repr(e)}")


def log_error(logger, msg: str):
    """Safe wrapper around logger.error that falls back to print if logging fails."""
    try:
        logger.error(msg)
    except Exception as e:
        print(f"[LOG-ERROR-FAILED] {msg} | {repr(e)}")


# ------ CONSTANTS FOR BUSINESS LOGIC (unchanged thresholds) ------

# Minimum number of products per PIM category needed to include that category
# in the training subset.
MIN_PRODUCTS_FOR_TRAINING = 10

# Conservative thresholds for global candidate rules:
# - MIN_KEYWORD_INSIDE_SUPPORT: keyword must appear at least this many times
#   inside a PIM category (Inside(C)) in the global training set.
# - MAX_KEYWORD_HARD_OUTSIDE: keyword may appear at most this many times
#   outside that category in the global training set.
MIN_KEYWORD_INSIDE_SUPPORT = 10
MAX_KEYWORD_HARD_OUTSIDE = 0

# Vendor-specific thresholds:
# - MIN_VENDOR_INSIDE_SUPPORT: keyword must appear at least this many times
#   in this vendor's products mapped to the PIM category.
# - MAX_VENDOR_OUTSIDE_SUPPORT: keyword may appear at most this many times
#   in this vendor's products mapped to other PIM categories.
MIN_VENDOR_INSIDE_SUPPORT = 3
MAX_VENDOR_OUTSIDE_SUPPORT = 0

# K2 (exclude-based) rule thresholds:
# - MIN_K2_EXCLUDE_SUPPORT: minimum occurrences of an exclude token outside the
#   target category to be considered as a valid exclude candidate.
# - MIN_K2_OUTSIDE_COVERAGE_RATIO: minimum ratio of outside occurrences that must
#   be covered by the exclude list.
# - MAX_K2_EXCLUDES_PER_RULE: maximum number of exclude tokens per K2 rule.
MIN_K2_EXCLUDE_SUPPORT = 3
MIN_K2_OUTSIDE_COVERAGE_RATIO = 0.2
MAX_K2_EXCLUDES_PER_RULE = 5


def normalize_training_record(rec: dict) -> dict:
    """
    Normalize a training record so that types are consistent across all records.

    This is a *technical safeguard* (for schema consistency) and does not change
    the business logic: all fields and values are preserved,
    only their Python types are normalised (e.g. strings vs. lists).
    """
    if not isinstance(rec, dict):
        return {}

    out = dict(rec)  # shallow copy

    # Normalise string-like fields to strings (if not None)
    str_fields = [
        "pim_category_id",
        "pim_category_name",
        "pim_category_path",
        "vendor_short_name",
        "vendor_category_id",
        "vendor_category_name",
        "vendor_category_path",
        "article_id",
        "description_short",
    ]
    for field in str_fields:
        if field in out and out[field] is not None:
            out[field] = str(out[field])
        elif field not in out:
            out[field] = None

    # Normalise keywords: always a list of strings
    raw_keywords = out.get("keywords")
    if raw_keywords is None:
        keywords = []
    elif isinstance(raw_keywords, list):
        # Trim whitespace from each keyword so training-side keywords match vendor-side trimming
        keywords = [
            str(k).strip() for k in raw_keywords
            if k is not None and str(k).strip() != ""
        ]
    else:
        kw_str = str(raw_keywords).strip()
        keywords = [kw_str] if kw_str != "" else []
    out["keywords"] = keywords

    # Normalise class_codes: always a list of {system, code}
    raw_class_codes = out.get("class_codes")
    norm_class_codes = []

    if raw_class_codes is None:
        norm_class_codes = []
    elif isinstance(raw_class_codes, list):
        for cc in raw_class_codes:
            if isinstance(cc, dict):
                norm_class_codes.append(
                    {
                        "system": cc.get("system"),
                        "code": str(cc.get("code")) if cc.get("code") is not None else None,
                    }
                )
            else:
                norm_class_codes.append(
                    {
                        "system": None,
                        "code": str(cc),
                    }
                )
    else:
        if isinstance(raw_class_codes, dict):
            norm_class_codes.append(
                {
                    "system": raw_class_codes.get("system"),
                    "code": str(raw_class_codes.get("code"))
                    if raw_class_codes.get("code") is not None
                    else None,
                }
            )
        else:
            norm_class_codes.append(
                {
                    "system": None,
                    "code": str(raw_class_codes),
                }
            )

    out["class_codes"] = norm_class_codes
    return out


def main():
    phase = "STARTUP"
    logger = None
    job = None

    try:
        print("GLUE SCRIPT: Parsing Glue arguments via getResolvedOptions()")
        args = getResolvedOptions(
            sys.argv,
            ["JOB_NAME", "INPUT_BUCKET", "OUTPUT_BUCKET", "vendor_name", "prepared_input_key", "prepared_output_prefix"],
        )
        job_name = args["JOB_NAME"]
        input_bucket = args["INPUT_BUCKET"]
        output_bucket = args["OUTPUT_BUCKET"]
        vendor_name = args["vendor_name"]
        prepared_input_key = args["prepared_input_key"]
        prepared_output_prefix = args["prepared_output_prefix"]

        print(
            "GLUE SCRIPT: Parsed arguments: "
            f"JOB_NAME={job_name}, INPUT_BUCKET={input_bucket}, "
            f"OUTPUT_BUCKET={output_bucket}, vendor_name={vendor_name}, "
            f"prepared_input_key={prepared_input_key}, "
            f"prepared_output_prefix={prepared_output_prefix}"
        )

        phase = "CONTEXT_INIT"
        print("GLUE SCRIPT: Creating SparkContext and GlueContext.")
        sc = SparkContext.getOrCreate()
        glue_context = GlueContext(sc)
        spark = glue_context.spark_session

        phase = "LOGGER_JOB_INIT"
        print("GLUE SCRIPT: Creating Glue Job and logger.")
        logger = glue_context.get_logger()
        job = Job(glue_context)
        job.init(job_name, args)
        print("GLUE SCRIPT: Glue Job initialised successfully.")

    except Exception as e:
        print("FATAL: Job failed during early phase:", phase, repr(e))
        traceback.print_exc()
        raise

    log_info(logger, "========== JOB START: Stable Training Dataset Update ==========")
    log_info(logger, f"JOB_NAME={job_name}")
    log_info(logger, f"INPUT_BUCKET={input_bucket}")
    log_info(logger, f"OUTPUT_BUCKET={output_bucket}")
    log_info(logger, f"vendor_name={vendor_name}")
    log_info(logger, f"prepared_input_key={prepared_input_key}")
    log_info(logger, f"prepared_output_prefix={prepared_output_prefix}")

    print("DEBUG: JOB START – Stable Training Dataset Update")
    print(f"DEBUG: JOB_NAME={job_name}")
    print(f"DEBUG: INPUT_BUCKET={input_bucket}")
    print(f"DEBUG: OUTPUT_BUCKET={output_bucket}")
    print(f"DEBUG: vendor_name={vendor_name}")
    print(f"DEBUG: prepared_input_key={prepared_input_key}")
    print(f"DEBUG: prepared_output_prefix={prepared_output_prefix}")

    s3_client = boto3.client("s3")
    current_step = "INIT"

    try:
        # =========================================================
        # STEP A: Determine input + stable dataset keys and existence
        # =========================================================
        current_step = "STEP A: Check input and Stable Training Dataset files"
        log_info(logger, current_step)

        input_key = (
            f"{prepared_output_prefix}"
            f"{vendor_name}_categoryMatchingProposals_oneVendor_to_onePim_match.json"
        )
        log_info(
            logger,
            "Expecting oneVendor_to_onePim_match at "
            f"s3://{input_bucket}/{input_key}",
        )
        print(
            "DEBUG: STEP A – expecting oneVendor_to_onePim_match at: "
            f"s3://{input_bucket}/{input_key}"
        )

        try:
            s3_client.head_object(Bucket=input_bucket, Key=input_key)
            log_info(
                logger,
                "STEP A: oneVendor_to_onePim_match file found on S3.",
            )
            print("DEBUG: STEP A – oneVendor_to_onePim_match file exists")
        except ClientError as ce:
            if ce.response["Error"]["Code"] in ("404", "NoSuchKey"):
                log_error(
                    logger,
                    "STEP A: oneVendor_to_onePim_match file not found. "
                    "Cannot proceed with training update.",
                )
                print(
                    "DEBUG: STEP A – oneVendor_to_onePim_match file not found; "
                    "failing job"
                )
                raise
            else:
                log_error(
                    logger,
                    "STEP A: Unexpected ClientError while checking "
                    "oneVendor_to_onePim_match file.",
                )
                log_error(logger, repr(ce))
                print(
                    "DEBUG: STEP A – unexpected ClientError during head_object; "
                    "failing job"
                )
                raise

        stable_key = "canonical_mappings/Category_Mapping_StableTrainingDataset.json"
        stable_training_key = (
            "canonical_mappings/Category_Mapping_StableTrainingDataset_training.json"
        )

        log_info(
            logger,
            "STEP A: Stable Training Dataset expected at "
            f"s3://{input_bucket}/{stable_key}",
        )
        print(
            "DEBUG: STEP A – expecting Stable Training Dataset at: "
            f"s3://{input_bucket}/{stable_key}"
        )

        # =========================================================
        # STEP B1: Read vendor oneVendor_to_onePim_match JSON
        # =========================================================
        current_step = "STEP B1: Read vendor oneVendor_to_onePim_match JSON"
        log_info(logger, current_step)

        log_info(
            logger,
            "Reading oneVendor_to_onePim_match JSON via boto3 "
            f"from s3://{input_bucket}/{input_key}",
        )
        print(
            "DEBUG: STEP B1 – reading oneVendor_to_onePim_match JSON from: "
            f"s3://{input_bucket}/{input_key}"
        )

        obj = s3_client.get_object(Bucket=input_bucket, Key=input_key)
        body_bytes = obj["Body"].read()
        try:
            match_data = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            log_error(logger, "STEP B1: Failed to parse input JSON as a dict.")
            log_error(logger, repr(e))
            print("DEBUG: STEP B1 – JSON parse failed; raising exception")
            raise

        if not isinstance(match_data, dict):
            log_error(
                logger,
                "STEP B1: Parsed JSON is not a dictionary keyed by vendor_category_id.",
            )
            print(
                "DEBUG: STEP B1 – match_data is not dict; type=",
                type(match_data).__name__,
            )
            raise ValueError("match_data is not a dict keyed by vendor_category_id")

        log_info(
            logger,
            f"STEP B1: Loaded match_data with {len(match_data)} vendor categories.",
        )
        print(
            "DEBUG: STEP B1 – vendor categories in match_data: "
            f"{len(match_data)}"
        )

        # =========================================================
        # STEP B2: Extract new training records from assignment_source
        # =========================================================
        current_step = "STEP B2: Extract new training records from assignment_source"
        log_info(logger, current_step)
        print(
            "DEBUG: STEP B2 – extracting products where assignment_source == "
            "'existing_category_match'"
        )

        new_training_records = []

        for vcat_id, vcat_record in match_data.items():
            if not isinstance(vcat_record, dict):
                continue

            pim_matches = vcat_record.get("pim_matches") or []
            if not isinstance(pim_matches, list):
                continue

            for m in pim_matches:
                if not isinstance(m, dict):
                    continue

                assignment_source = m.get("assignment_source")
                if assignment_source != "existing_category_match":
                    continue

                pim_category_id = m.get("pim_category_id")
                pim_category_name = m.get("pim_category_name")
                pim_category_path = m.get("pim_category_path")

                products = m.get("products") or []
                if not isinstance(products, list):
                    continue

                for p in products:
                    if not isinstance(p, dict):
                        continue
                    article_id = p.get("article_id")
                    if article_id is None or str(article_id).strip() == "":
                        log_warning(
                            logger,
                            "STEP B2: Skipping product without valid article_id "
                            f"in vendor_category_id={vcat_id}",
                        )
                        print(
                            "DEBUG: STEP B2 – skipped product without valid "
                            f"article_id in vendor_category_id={vcat_id}"
                        )
                        continue

                    description_short = p.get("description_short")

                    raw_keywords = p.get("keywords")
                    if raw_keywords is None:
                        keywords = []
                    elif isinstance(raw_keywords, list):
                        keywords = [
                            str(k)
                            for k in raw_keywords
                            if k is not None and str(k).strip() != ""
                        ]
                    else:
                        kw_str = str(raw_keywords)
                        keywords = [kw_str] if kw_str.strip() != "" else []

                    class_codes = p.get("class_codes")

                    record = {
                        "pim_category_id": pim_category_id,
                        "pim_category_name": pim_category_name,
                        "pim_category_path": pim_category_path,
                        "vendor_short_name": vendor_name,
                        "vendor_category_id": vcat_id,
                        "vendor_category_name": vcat_record.get("vendor_category_name"),
                        "vendor_category_path": vcat_record.get("vendor_category_path"),
                        "article_id": article_id,
                        "description_short": description_short,
                        "keywords": keywords,
                        "class_codes": class_codes,
                    }
                    new_training_records.append(record)

        log_info(
            logger,
            f"STEP B2: Extracted {len(new_training_records)} new training records "
            "from existing_category_match entries.",
        )
        print(
            "DEBUG: STEP B2 – new_training_records count: "
            f"{len(new_training_records)}"
        )

        if not new_training_records:
            log_info(
                logger,
                "STEP B2: No existing_category_match products found for this vendor. "
                "Stable Training Dataset will not be extended, but existing training "
                "data (if present) will still be used for training subset and "
                "rule proposal generation for this vendor.",
            )
            print(
                "DEBUG: STEP B2 – no new_training_records; will reuse existing "
                "Stable Training Dataset only"
            )

        # =========================================================
        # STEP B3: Merge with existing Stable Training Dataset
        # =========================================================
        current_step = "STEP B3: Merge new records into Stable Training Dataset"
        log_info(logger, current_step)
        print(
            "DEBUG: STEP B3 – merging new training records into Stable Training Dataset"
        )

        existing_records = []
        try:
            stable_obj = s3_client.get_object(Bucket=input_bucket, Key=stable_key)
            stable_bytes = stable_obj["Body"].read()
            try:
                existing_records = json.loads(stable_bytes.decode("utf-8"))
            except Exception as e:
                log_error(
                    logger,
                    "STEP B3: Failed to parse Stable Training Dataset JSON. "
                    "Cannot safely merge.",
                )
                log_error(logger, repr(e))
                print(
                    "DEBUG: STEP B3 – failed to parse existing Stable Dataset JSON; "
                    "raising exception"
                )
                raise
            if not isinstance(existing_records, list):
                log_error(
                    logger,
                    "STEP B3: Stable Training Dataset JSON is not a list of records.",
                )
                print(
                    "DEBUG: STEP B3 – existing_records is not list; type=",
                    type(existing_records).__name__,
                )
                raise ValueError("Stable Training Dataset JSON must be a list")
        except ClientError as ce:
            if ce.response["Error"]["Code"] in ("404", "NoSuchKey"):
                log_warning(
                    logger,
                    "STEP B3: Stable Training Dataset does not yet exist. "
                    "A new one will be created from this vendor's records.",
                )
                print(
                    "DEBUG: STEP B3 – Stable Training Dataset not found; "
                    "starting from empty list"
                )
                existing_records = []
            else:
                log_error(
                    logger,
                    "STEP B3: Unexpected error while accessing Stable Training "
                    f"Dataset: {repr(ce)}",
                )
                print(
                    "DEBUG: STEP B3 – unexpected ClientError while accessing "
                    "Stable Training Dataset; raising exception"
                )
                raise

        log_info(
            logger,
            f"STEP B3: Loaded {len(existing_records)} existing training records "
            "from Stable Training Dataset (before merge).",
        )
        print(
            "DEBUG: STEP B3 – existing_records (before merge): "
            f"{len(existing_records)}"
        )

        merged_by_key = {}

        for rec in existing_records:
            if not isinstance(rec, dict):
                continue
            vsn = str(rec.get("vendor_short_name") or "").strip()
            aid = str(rec.get("article_id") or "").strip()
            if not vsn or not aid:
                continue
            merged_by_key[(vsn, aid)] = rec

        for rec in new_training_records:
            vsn = str(rec.get("vendor_short_name") or "").strip()
            aid = str(rec.get("article_id") or "").strip()
            if not vsn or not aid:
                continue
            merged_by_key[(vsn, aid)] = rec

        final_records = list(merged_by_key.values())

        log_info(
            logger,
            f"STEP B3: After merge, Stable Training Dataset has "
            f"{len(final_records)} records.",
        )
        print(
            "DEBUG: STEP B3 – final_records (after merge): "
            f"{len(final_records)}"
        )

        # =========================================================
        # STEP B4: Write back Stable Training Dataset
        # =========================================================
        current_step = "STEP B4: Write back Stable Training Dataset"
        log_info(logger, current_step)
        print("DEBUG: STEP B4 – writing Stable Training Dataset to S3")

        stable_body = json.dumps(final_records, indent=2, ensure_ascii=False)
        s3_client.put_object(
            Bucket=input_bucket,
            Key=stable_key,
            Body=stable_body.encode("utf-8"),
        )

        log_info(logger, "STEP B4: Stable Training Dataset successfully updated.")
        print("DEBUG: STEP B4 – Stable Training Dataset write completed")

        # =========================================================
        # STEP C: Derive training subset with min product threshold
        # =========================================================
        current_step = "STEP C: Derive training subset"
        log_info(logger, current_step)
        print(
            "DEBUG: STEP C – deriving training subset with "
            f"MIN_PRODUCTS_FOR_TRAINING={MIN_PRODUCTS_FOR_TRAINING}"
        )

        category_counts = {}
        for rec in final_records:
            cid = rec.get("pim_category_id")
            if cid is None:
                continue
            cid_str = str(cid).strip()
            if not cid_str:
                continue
            category_counts[cid_str] = category_counts.get(cid_str, 0) + 1

        log_info(
            logger,
            "STEP C: Computed product counts for "
            f"{len(category_counts)} distinct pim_category_id values.",
        )
        print(
            "DEBUG: STEP C – distinct pim_category_id count: "
            f"{len(category_counts)}"
        )

        usable_categories = [
            cid for cid, cnt in category_counts.items()
            if cnt >= MIN_PRODUCTS_FOR_TRAINING
        ]

        log_info(
            logger,
            f"STEP C: {len(usable_categories)} PIM categories meet the minimum "
            f"threshold of {MIN_PRODUCTS_FOR_TRAINING} products for training.",
        )
        print(
            "DEBUG: STEP C – usable_categories (>= "
            f"{MIN_PRODUCTS_FOR_TRAINING} products): "
            f"{len(usable_categories)} -> {usable_categories}"
        )

        usable_set = set(usable_categories)
        training_records = [
            normalize_training_record(rec)
            for rec in final_records
            if str(rec.get("pim_category_id") or "").strip() in usable_set
        ]

        print(
            "DEBUG: STEP C – training_records count after filtering: "
            f"{len(training_records)}"
        )

        previous_training_records = []
        if existing_records:
            prev_category_counts = {}
            for rec in existing_records:
                cid = rec.get("pim_category_id")
                if cid is None:
                    continue
                cid_str = str(cid).strip()
                if not cid_str:
                    continue
                prev_category_counts[cid_str] = prev_category_counts.get(cid_str, 0) + 1

            log_info(
                logger,
                "STEP C: Computed product counts for previous Stable Training Dataset "
                f"with {len(prev_category_counts)} distinct pim_category_id values.",
            )
            print(
                "DEBUG: STEP C – previous training distinct pim_category_id count: "
                f"{len(prev_category_counts)}"
            )

            prev_usable_categories = [
                cid for cid, cnt in prev_category_counts.items()
                if cnt >= MIN_PRODUCTS_FOR_TRAINING
            ]

            log_info(
                logger,
                "STEP C: Previous Stable Training Dataset – "
                f"{len(prev_usable_categories)} PIM categories meet the minimum "
                f"threshold of {MIN_PRODUCTS_FOR_TRAINING} products for training.",
            )
            print(
                "DEBUG: STEP C – previous usable_categories (>= "
                f"{MIN_PRODUCTS_FOR_TRAINING} products): "
                f"{len(prev_usable_categories)} -> {prev_usable_categories}"
            )

            prev_usable_set = set(prev_usable_categories)
            previous_training_records = [
                normalize_training_record(rec)
                for rec in existing_records
                if str(rec.get("pim_category_id") or "").strip() in prev_usable_set
            ]

            print(
                "DEBUG: STEP C – previous_training_records count after filtering: "
                f"{len(previous_training_records)}"
            )
        else:
            previous_training_records = []

        if training_records:
            sample = training_records[0]
            print(
                "DEBUG: STEP C – sample training_record type:",
                type(sample).__name__,
            )
            try:
                sample_keys = list(sample.keys())
                print("DEBUG: STEP C – sample training_record keys:", sample_keys)
            except Exception as e_keys:
                print(
                    "DEBUG: STEP C – could not read sample training_record keys:",
                    repr(e_keys),
                )
        else:
            print("DEBUG: STEP C – training_records is empty list")

        # =========================================================
        # STEP D: Prepare training subset for KEYWORD statistics
        # =========================================================
        current_step = "STEP D: Generate conservative KEYWORD-based rule proposals"
        log_info(logger, current_step)

        log_info(
            logger,
            "STEP D: Generating conservative KEYWORD-based rule proposals "
            "from training subset and this vendor's matches...",
        )
        print(
            "DEBUG: STEP D – starting rule proposal generation; "
            f"training_records={len(training_records)}"
        )

        if not training_records:
            log_info(
                logger,
                "STEP D: Training subset is empty (no categories above threshold). "
                "Skipping rule proposal generation. "
                "No rule proposal file will be created.",
            )
            log_info(logger, "========== JOB END (SUCCESS) ==========")
            print(
                "DEBUG: EARLY EXIT – STEP D: training subset empty; "
                "no rule proposals will be generated"
            )
            job.commit()
            return

        # ---------- STEP D1: Create DataFrame from training_records ----------
        current_step = "STEP D1: Create DataFrame from training_records"
        log_info(logger, current_step)
        print("DEBUG: STEP D1 – creating DataFrame from training_records")
        print(f"DEBUG: STEP D1 – training_records length: {len(training_records)}")

        if training_records:
            sample = training_records[0]
            print("DEBUG: STEP D1 – sample record type:", type(sample).__name__)
            try:
                sample_keys = list(sample.keys())
                print("DEBUG: STEP D1 – sample record keys:", sample_keys)
            except Exception as e_keys:
                print(
                    "DEBUG: STEP D1 – could not read sample record keys:",
                    repr(e_keys),
                )
            try:
                print("DEBUG: STEP D1 – sample record:", json.dumps(sample)[:500])
            except Exception as e_json:
                print(
                    "DEBUG: STEP D1 – could not JSON-serialise sample record:",
                    repr(e_json),
                )

        training_schema = StructType([
            StructField("pim_category_id", StringType(), True),
            StructField("pim_category_name", StringType(), True),
            StructField("pim_category_path", StringType(), True),
            StructField("vendor_short_name", StringType(), True),
            StructField("vendor_category_id", StringType(), True),
            StructField("vendor_category_name", StringType(), True),
            StructField("vendor_category_path", StringType(), True),
            StructField("article_id", StringType(), True),
            StructField("description_short", StringType(), True),
            StructField("keywords", ArrayType(StringType()), True),
            StructField(
                "class_codes",
                ArrayType(
                    StructType([
                        StructField("system", StringType(), True),
                        StructField("code", StringType(), True),
                    ])
                ),
                True,
            ),
        ])

        try:
            df_train = glue_context.spark_session.createDataFrame(
                training_records,
                schema=training_schema,
            )
        except Exception as e_df:
            log_error(
                logger,
                "STEP D1: Failed to create DataFrame from training_records. "
                "No rule proposals will be generated.",
            )
            log_error(logger, repr(e_df))
            print(
                "DEBUG: STEP D1 – DataFrame creation from training_records failed; "
                "no rule proposals will be generated"
            )
            job.commit()
            return

        for col_name in ["pim_category_id", "article_id", "keywords"]:
            if col_name not in df_train.columns:
                log_warning(
                    logger,
                    f"STEP D1: Column '{col_name}' missing in training DataFrame; "
                    "adding it as NULL.",
                )
                df_train = df_train.withColumn(col_name, F.lit(None))
                print(
                    f"DEBUG: STEP D1 – column '{col_name}' missing "
                    "in df_train; added as NULL"
                )

        if previous_training_records:
            current_step = "STEP D1: Prepare previous training KEYWORD pairs"
            log_info(logger, current_step)
            print("DEBUG: STEP D1 – building previous training KEYWORD pairs from previous_training_records")

            try:
                # Build a minimal, flat list of (pim_category_id, keyword) rows in Python to avoid
                # Spark schema inference issues on nested record structures.
                prev_kw_rows = []
                for rec in previous_training_records:
                    if not isinstance(rec, dict):
                        continue
                    pim_category_id = rec.get("pim_category_id")
                    raw_keywords = rec.get("keywords") or []
                    if raw_keywords is None:
                        continue
                    # Ensure keywords is iterable
                    if not isinstance(raw_keywords, list):
                        raw_keywords = [raw_keywords]
                    for kw in raw_keywords:
                        if kw is None:
                            continue
                        kw_str = str(kw).strip()
                        if not kw_str:
                            continue
                        prev_kw_rows.append({
                            "pim_category_id": str(pim_category_id) if pim_category_id is not None else None,
                            "keyword": kw_str,
                        })

                if prev_kw_rows:
                    prev_schema = StructType([
                        StructField("pim_category_id", StringType(), True),
                        StructField("keyword", StringType(), True),
                    ])
                    df_prev_kw = glue_context.spark_session.createDataFrame(prev_kw_rows, schema=prev_schema)
                    # Deduplicate to match previous behavior
                    df_prev_kw = df_prev_kw.dropDuplicates(["pim_category_id", "keyword"])
                    log_info(logger, f"STEP D1: Built previous keyword pairs DataFrame with {df_prev_kw.count()} rows")
                    print(f"DEBUG: STEP D1 – previous keyword pairs count: {df_prev_kw.count()}")
                else:
                    df_prev_kw = glue_context.spark_session.createDataFrame([], schema="pim_category_id string, keyword string")
                    log_info(logger, "STEP D1: No previous keyword pairs found in previous_training_records")
                    print("DEBUG: STEP D1 – no previous keyword pairs found in previous_training_records")

            except Exception as e_prev:
                # Log the actual exception and fall back to empty DataFrame (preserve existing behavior)
                log_warning(logger, f"STEP D1: Could not build previous-training keyword pairs; proceeding without previous training info. Exception: {repr(e_prev)}")
                print("DEBUG: STEP D1 – DataFrame creation from previous_training_records failed; proceeding without previous training info")
                df_prev_kw = glue_context.spark_session.createDataFrame([], schema="pim_category_id string, keyword string")
        else:
            df_prev_kw = glue_context.spark_session.createDataFrame([], schema="pim_category_id string, keyword string")

        # ---------- STEP D2: Explode KEYWORD terms from training subset ----------
        current_step = "STEP D2: Explode KEYWORD terms from training subset"
        log_info(logger, current_step)
        print("DEBUG: STEP D2 – exploding KEYWORD terms from training subset")

        df_kw_train = (
            df_train
            .select(
                F.col("pim_category_id").alias("pim_category_id"),
                F.col("article_id").alias("article_id"),
                F.explode_outer("keywords").alias("keyword"),
            )
            .where(
                F.col("keyword").isNotNull()
                & (F.trim(F.col("keyword")) != F.lit(""))
            )
        )

        current_step = "STEP D2: Compute inside_count and total training counts"
        log_info(logger, current_step)
        print("DEBUG: STEP D2 – computing inside_count and train_total_count")

        df_inside = (
            df_kw_train
            .groupBy("pim_category_id", "keyword")
            .agg(F.count(F.lit(1)).alias("inside_count"))
        )

        df_total_train = (
            df_kw_train
            .groupBy("keyword")
            .agg(F.count(F.lit(1)).alias("train_total_count"))
        )

        df_stats = (
            df_inside
            .join(df_total_train, on="keyword", how="left")
            .withColumn(
                "hard_outside_count",
                F.col("train_total_count") - F.col("inside_count"),
            )
        )

        log_info(
            logger,
            "STEP D2: Completed computation of inside_count, train_total_count "
            "and hard_outside_count.",
        )
        print(
            "DEBUG: STEP D2 – completed inside_count/train_total_count/"
            "hard_outside_count computation"
        )

        # =========================================================
        # STEP D3: Build vendor KEYWORD statistics
        # =========================================================
        current_step = "STEP D3: Build vendor KEYWORD statistics"
        log_info(logger, current_step)
        print("DEBUG: STEP D3 – building vendor KEYWORD statistics from match_data")

        vendor_kw_rows = []
        for vcat_key, vcat_record in match_data.items():
            if not isinstance(vcat_record, dict):
                continue
            pim_matches = vcat_record.get("pim_matches") or []
            if not isinstance(pim_matches, list):
                continue

            for m in pim_matches:
                if not isinstance(m, dict):
                    continue

                pim_category_id = m.get("pim_category_id")
                products = m.get("products") or []
                if not isinstance(products, list):
                    continue

                for p in products:
                    if not isinstance(p, dict):
                        continue
                    article_id = p.get("article_id")
                    raw_keywords = p.get("keywords")
                    if raw_keywords is None:
                        kws = []
                    elif isinstance(raw_keywords, list):
                        kws = raw_keywords
                    else:
                        kws = [raw_keywords]

                    for kw in kws:
                        if kw is None:
                            continue
                        kw_str = str(kw).strip()
                        if not kw_str:
                            continue
                        vendor_kw_rows.append(
                            {
                                "pim_category_id": str(pim_category_id)
                                if pim_category_id is not None
                                else None,
                                "article_id": str(article_id)
                                if article_id is not None
                                else None,
                                "keyword": kw_str,
                            }
                        )

        if vendor_kw_rows:
            df_kw_vendor = glue_context.spark_session.createDataFrame(
                vendor_kw_rows,
                schema=StructType([
                    StructField("pim_category_id", StringType(), True),
                    StructField("article_id", StringType(), True),
                    StructField("keyword", StringType(), True),
                ]),
            )

            df_vendor_inside = (
                df_kw_vendor
                .groupBy("pim_category_id", "keyword")
                .agg(F.count(F.lit(1)).alias("vendor_inside_count"))
            )

            df_vendor_total = (
                df_kw_vendor
                .groupBy("keyword")
                .agg(F.count(F.lit(1)).alias("vendor_total_count"))
            )

            log_info(
                logger,
                "STEP D3: Built vendor keyword statistics "
                "(vendor_inside_count, vendor_total_count).",
            )
            print(
                "DEBUG: STEP D3 – built vendor_inside_count/"
                "vendor_total_count "
                "DataFrames"
            )
        else:
            log_info(
                logger,
                "STEP D3: No keyword entries found in vendor oneVendor_to_onePim "
                "data. Rule proposals will be based only on global training statistics.",
            )
            print(
                "DEBUG: STEP D3 – no keyword entries found in vendor data; "
                "vendor stats will be empty"
            )
            df_vendor_inside = glue_context.spark_session.createDataFrame(
                [], schema="pim_category_id string, keyword string, vendor_inside_count long"
            )
            df_vendor_total = glue_context.spark_session.createDataFrame(
                [], schema="keyword string, vendor_total_count long"
            )

        current_step = "STEP D3: Join training and vendor statistics"
        log_info(logger, current_step)
        print("DEBUG: STEP D3 – joining training and vendor keyword statistics")

        df_stats = (
            df_stats
            .join(
                df_vendor_inside,
                on=["pim_category_id", "keyword"],
                how="left",
            )
            .join(
                df_vendor_total,
                on="keyword",
                how="left",
            )
        )

        df_stats = df_stats.fillna(
            {"vendor_inside_count": 0, "vendor_total_count": 0}
        )

        df_stats = df_stats.withColumn(
            "vendor_outside_count",
            F.col("vendor_total_count") - F.col("vendor_inside_count"),
        )

        log_info(
            logger,
            "STEP D3: Completed join of training and vendor statistics and "
            "computed vendor_outside_count.",
        )
        print(
            "DEBUG: STEP D3 – completed join of training and vendor statistics "
            "and vendor_outside_count computation"
        )

        # =========================================================
        # STEP D5: Apply thresholds and derive rule status for each (category, keyword)
        # =========================================================
        current_step = "STEP D5: Apply thresholds and derive rule status"
        log_info(logger, current_step)
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
        if conservative_count == 0:
            log_info(
                logger,
                "STEP D5: No KEYWORD terms met the very conservative thresholds "
                "for this vendor. Rule proposals will therefore likely be empty "
                "(no candidate KEYWORD rules above thresholds).",
            )
            print(
                "DEBUG: STEP D5 – no KEYWORD terms met conservative thresholds; "
                "rule proposal set may be empty"
            )
        else:
            log_info(
                logger,
                f"STEP D5: Found at least one KEYWORD term ({conservative_count}+) "
                "that meets the very conservative thresholds.",
            )
            print(
                "DEBUG: STEP D5 – at least one candidate KEYWORD term passes "
                "conservative filters"
            )

        current_step = "STEP D5: Join with previous training pairs for new-rule detection"
        log_info(logger, current_step)
        print(
            "DEBUG: STEP D5 – joining df_stats with previous training KEYWORD pairs"
        )

        df_stats = (
            df_stats
            .join(
                df_prev_kw.withColumn("known_in_previous_training", F.lit(True)),
                on=["pim_category_id", "keyword"],
                how="left",
            )
            .fillna({"known_in_previous_training": False})
        )

        df_stats = (
            df_stats
            .withColumn(
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
        log_info(logger, current_step)
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

        # Keep only rules that actually meet the global candidate thresholds
        # and have a meaningful vendor_status (new / supported / violated / not_impacted).
        df_output = df_stats.where(
            F.col("global_candidate") & F.col("vendor_status").isNotNull()
        )

        current_step = "STEP D5: Collect rule status rows to driver"
        log_info(logger, current_step)
        print("DEBUG: STEP D5 – collecting full rule status rows to driver")

        rows = df_output.collect()
        rule_proposals = []

        for r in rows:
            pim_category_id = r["pim_category_id"]
            pim_category_id_str = str(pim_category_id) if pim_category_id is not None else None
            pim_category_name = category_name_map.get(pim_category_id_str)
            keyword = r["keyword"]

            inside_count = int(r["inside_count"]) if r["inside_count"] is not None else 0
            train_total_count = int(r["train_total_count"]) if r["train_total_count"] is not None else 0
            hard_outside_count = int(r["hard_outside_count"]) if r["hard_outside_count"] is not None else 0
            vendor_inside_count = int(r["vendor_inside_count"]) if r["vendor_inside_count"] is not None else 0
            vendor_total_count = int(r["vendor_total_count"]) if r["vendor_total_count"] is not None else 0
            vendor_outside_count = int(r["vendor_outside_count"]) if r["vendor_outside_count"] is not None else 0
            vendor_status = r["vendor_status"] if r["vendor_status"] is not None else None

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

        # =========================================================
        # K2 Mapping Method: Generate exclude-based rules
        # =========================================================
        current_step = "STEP K2: Generate K2 exclude-based keyword rules"
        log_info(logger, current_step)
        print("DEBUG: STEP K2 – starting K2 rule generation")

        def _is_meaningful_token(tok: str) -> bool:
            if not tok:
                return False
            # require length >= 4
            if len(tok) < 4:
                return False
            # require at least one letter (Unicode-aware)
            if not any(ch.isalpha() for ch in tok):
                return False
            # reject pure-digit tokens
            if re.fullmatch(r'[0-9]+', tok):
                return False
            # reject digit-only tokens with separators (likely article numbers)
            if re.fullmatch(r'[0-9\.\-_/]+', tok):
                return False
            return True

        # Step 1: Build training_kw_occurrences (case-insensitive)
        training_kw_occurrences = {}
        for rec in training_records:
            pim_category_id_raw = rec.get("pim_category_id")
            # Normalize pim_category_id to string for consistent comparisons
            pim_category_id = str(pim_category_id_raw) if pim_category_id_raw is not None else None
            raw_keywords = rec.get("keywords") or []
            if not isinstance(raw_keywords, list):
                continue
            # Build lowercased set of keywords for this record
            keywords_set_lower = set()
            for kw in raw_keywords:
                if kw is None:
                    continue
                kw_lower = str(kw).strip().lower()
                if kw_lower:
                    keywords_set_lower.add(kw_lower)
            
            if not keywords_set_lower:
                continue
            
            # Add to occurrence index for each keyword
            for kw_lower in keywords_set_lower:
                if kw_lower not in training_kw_occurrences:
                    training_kw_occurrences[kw_lower] = []
                training_kw_occurrences[kw_lower].append({
                    "pim_category_id": pim_category_id,
                    "keywords_set_lower": keywords_set_lower
                })

        # Step 2: Build vendor_kw_occurrences from match_data (case-insensitive)
        vendor_kw_occurrences = {}
        for vcat_key, vcat_record in match_data.items():
            if not isinstance(vcat_record, dict):
                continue
            pim_matches = vcat_record.get("pim_matches") or []
            if not isinstance(pim_matches, list):
                continue

            for m in pim_matches:
                if not isinstance(m, dict):
                    continue
                pim_category_id_raw = m.get("pim_category_id")
                # Normalize pim_category_id to string for consistent comparisons
                pim_category_id = str(pim_category_id_raw) if pim_category_id_raw is not None else None
                products = m.get("products") or []
                if not isinstance(products, list):
                    continue

                for p in products:
                    if not isinstance(p, dict):
                        continue
                    raw_keywords = p.get("keywords")
                    if raw_keywords is None:
                        kws = []
                    elif isinstance(raw_keywords, list):
                        kws = raw_keywords
                    else:
                        kws = [raw_keywords]

                    # Build lowercased set of keywords for this product
                    keywords_set_lower = set()
                    for kw in kws:
                        if kw is None:
                            continue
                        kw_lower = str(kw).strip().lower()
                        if kw_lower:
                            keywords_set_lower.add(kw_lower)
                    
                    if not keywords_set_lower:
                        continue
                    
                    # Add to occurrence index for each keyword
                    for kw_lower in keywords_set_lower:
                        if kw_lower not in vendor_kw_occurrences:
                            vendor_kw_occurrences[kw_lower] = []
                        vendor_kw_occurrences[kw_lower].append({
                            "pim_category_id": pim_category_id,
                            "keywords_set_lower": keywords_set_lower
                        })

        # Step 3: Select K2 candidate rows from df_stats
        k2_candidates = df_stats.where(
            (F.col("inside_count") >= F.lit(MIN_KEYWORD_INSIDE_SUPPORT))
            & (F.col("hard_outside_count") > F.lit(MAX_KEYWORD_HARD_OUTSIDE))
        ).collect()

        log_info(logger, f"STEP K2: Found {len(k2_candidates)} K2 candidate rows")
        print(f"DEBUG: STEP K2 – found {len(k2_candidates)} K2 candidates")

        # Step 4: Process each K2 candidate
        k2_rules_count = 0
        for r in k2_candidates:
            try:
                pim_category_id = r["pim_category_id"]
                pim_category_id_norm = (
                    str(pim_category_id) if pim_category_id is not None else None
                )
                keyword = r["keyword"]
                keyword_lower = keyword.strip().lower()

                # Get occurrences for this keyword
                occ_list = training_kw_occurrences.get(keyword_lower, [])
                if not occ_list:
                    continue

                # Partition into inside and outside sets
                inside_sets = []
                outside_sets = []
                for occ in occ_list:
                    occ_pim_category_id_norm = (
                        str(occ["pim_category_id"]) if occ["pim_category_id"] is not None else None
                    )
                    if occ_pim_category_id_norm == pim_category_id_norm:
                        inside_sets.append(occ["keywords_set_lower"])
                    else:
                        outside_sets.append(occ["keywords_set_lower"])

                # Compute inside_tokens and outside_tokens
                inside_tokens = set()
                for s in inside_sets:
                    inside_tokens.update(s)
                
                outside_tokens = set()
                for s in outside_sets:
                    outside_tokens.update(s)

                # Compute exclude_candidates (only tokens appearing outside but not inside)
                exclude_candidates_raw = outside_tokens - inside_tokens
                # lexical filter: keep only meaningful tokens (letters, len>=4, not numeric codes)
                exclude_candidates_raw = {tok for tok in exclude_candidates_raw if _is_meaningful_token(tok)}
                if not exclude_candidates_raw:
                    # nothing meaningful to exclude -> skip K2 candidate
                    continue

                # Apply MIN_K2_EXCLUDE_SUPPORT: filter exclude candidates by support count
                # Count how many times each exclude candidate appears in outside_sets
                exclude_support_counts = {}
                for token in exclude_candidates_raw:
                    count = sum(1 for s in outside_sets if token in s)
                    exclude_support_counts[token] = count
                
                # Keep only candidates with sufficient support
                exclude_candidates = {
                    token for token in exclude_candidates_raw
                    if exclude_support_counts.get(token, 0) >= MIN_K2_EXCLUDE_SUPPORT
                }
                
                if not exclude_candidates:
                    continue

                # Apply MAX_K2_EXCLUDES_PER_RULE: limit number of exclude tokens
                if len(exclude_candidates) > MAX_K2_EXCLUDES_PER_RULE:
                    # Select top N tokens by support count
                    sorted_candidates = sorted(
                        exclude_candidates,
                        key=lambda t: exclude_support_counts.get(t, 0),
                        reverse=True
                    )
                    exclude_candidates = set(sorted_candidates[:MAX_K2_EXCLUDES_PER_RULE])

                # Validate coverage: require that exclude_candidates cover at least
                # MIN_K2_OUTSIDE_COVERAGE_RATIO of outside occurrences
                covered_outside_sets = sum(1 for s in outside_sets if s & exclude_candidates)
                total_outside_sets = len(outside_sets)
                
                if total_outside_sets > 0:
                    coverage_ratio = covered_outside_sets / total_outside_sets
                    if coverage_ratio < MIN_K2_OUTSIDE_COVERAGE_RATIO:
                        continue
                else:
                    # No outside sets to cover, skip this candidate
                    continue

                # Compute vendor stats under K2 semantics
                vendor_occ_list = vendor_kw_occurrences.get(keyword_lower, [])
                vendor_total_k2 = 0
                vendor_inside_k2 = 0
                vendor_outside_k2 = 0

                for vocc in vendor_occ_list:
                    # If keyword set intersects exclude_candidates, exclude this occurrence
                    if vocc["keywords_set_lower"] & exclude_candidates:
                        continue
                    
                    # Count this occurrence
                    vendor_total_k2 += 1
                    vocc_pim_category_id_norm = (
                        str(vocc["pim_category_id"]) if vocc["pim_category_id"] is not None else None
                    )
                    if vocc_pim_category_id_norm == pim_category_id_norm:
                        vendor_inside_k2 += 1
                    else:
                        vendor_outside_k2 += 1

                # Decide vendor_status_k2
                if vendor_total_k2 == 0:
                    vendor_status_k2 = "not_impacted"
                elif vendor_inside_k2 >= MIN_VENDOR_INSIDE_SUPPORT and vendor_outside_k2 <= MAX_VENDOR_OUTSIDE_SUPPORT:
                    # Check if this is a known rule from previous training
                    known_in_previous = r["known_in_previous_training"] if r["known_in_previous_training"] is not None else False
                    vendor_status_k2 = "new" if not known_in_previous else "supported"
                else:
                    # Vendor violates K2 semantics, skip this candidate
                    continue

                # Build K2 rule object
                pim_category_id_str = str(pim_category_id) if pim_category_id is not None else None
                pim_category_name = category_name_map.get(pim_category_id_str)
                inside_count = int(r["inside_count"]) if r["inside_count"] is not None else 0

                k2_rule_obj = {
                    "pim_category_id": pim_category_id,
                    "pim_category_name": pim_category_name,
                    "field_name": "KEYWORD",
                    "operator": "contains_any_exclude_any",
                    "values_include": [keyword.strip()],
                    "values_exclude": sorted(list(exclude_candidates)),
                    "stats": {
                        "inside_count": inside_count,
                        "train_total_count": inside_count,
                        "hard_outside_count": 0,
                        "vendor_inside_count": vendor_inside_k2,
                        "vendor_total_count": vendor_total_k2,
                        "vendor_outside_count": vendor_outside_k2,
                        "vendor_status": vendor_status_k2,
                    },
                }
                rule_proposals.append(k2_rule_obj)
                k2_rules_count += 1

                # Log accepted K2 rule
                log_info(
                    logger,
                    f"STEP K2: Accepted K2 rule - category={pim_category_id}, "
                    f"keyword={keyword}, exclude_count={len(exclude_candidates)}, "
                    f"vendor_status={vendor_status_k2}"
                )

            except Exception as e_k2:
                log_warning(
                    logger,
                    f"STEP K2: Failed to process K2 candidate "
                    f"(category={r.get('pim_category_id')}, keyword={r.get('keyword')}): {repr(e_k2)}"
                )
                continue

        log_info(logger, f"STEP K2: Appended {k2_rules_count} K2 rules to rule_proposals")
        print(f"DEBUG: STEP K2 – appended {k2_rules_count} K2 rules")

        log_info(
            logger,
            f"STEP D5: Generated {len(rule_proposals)} rule-status entries "
            f"for vendor={vendor_name}.",
        )
        print(
            "DEBUG: STEP D5 – generated rule-status entries: "
            f"{len(rule_proposals)}"
        )

        # =========================================================
        # STEP D6: Write rule proposals JSON to S3
        # =========================================================
        current_step = "STEP D6: Write rule proposals JSON to S3"
        log_info(logger, current_step)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rule_key = (
            "canonical_mappings/"
            f"Category_Mapping_RuleProposals_{vendor_name}_{ts}.json"
        )

        body_rules = json.dumps(rule_proposals, indent=2, ensure_ascii=False)
        log_info(
            logger,
            "Writing rule proposals to "
            f"s3://{input_bucket}/{rule_key}",
        )
        print(
            "DEBUG: STEP D6 – writing rule proposals to: "
            f"s3://{input_bucket}/{rule_key}"
        )

        s3_client.put_object(
            Bucket=input_bucket,
            Key=rule_key,
            Body=body_rules.encode("utf-8"),
        )

        log_info(logger, "STEP D6: Rule proposals successfully written to S3.")

        # =========================================================
        # STEP E: Update Category Mapping Reference and write change log
        # =========================================================
        current_step = "STEP E: Update Category Mapping Reference and write change log"
        log_info(logger, current_step)
        print("DEBUG: STEP E – updating Category_Mapping_Reference and writing change log")

        # Helper: find latest existing Category_Mapping_Reference_*.json
        prefix_ref = "canonical_mappings/Category_Mapping_Reference_"
        latest_ref_key = None

        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=input_bucket, Prefix=prefix_ref)
            for page in page_iterator:
                contents = page.get("Contents")
                if not contents:
                    continue
                for obj in contents:
                    key = obj.get("Key")
                    if not key or not key.endswith(".json"):
                        continue
                    if (latest_ref_key is None) or (key > latest_ref_key):
                        latest_ref_key = key
        except Exception as e_list:
            log_warning(
                logger,
                "STEP E: Failed to list existing Category_Mapping_Reference files; "
                "treating as none.",
            )
            log_warning(logger, repr(e_list))
            latest_ref_key = None

        # Load existing reference (if any)
        old_reference = []
        if latest_ref_key:
            try:
                log_info(
                    logger,
                    f"STEP E: Loading existing Category_Mapping_Reference from "
                    f"s3://{input_bucket}/{latest_ref_key}",
                )
                print(
                    "DEBUG: STEP E – loading existing reference from: "
                    f"s3://{input_bucket}/{latest_ref_key}"
                )
                ref_obj = s3_client.get_object(Bucket=input_bucket, Key=latest_ref_key)
                ref_bytes = ref_obj["Body"].read()
                old_reference = json.loads(ref_bytes.decode("utf-8"))
                if not isinstance(old_reference, list):
                    log_warning(
                        logger,
                        "STEP E: Existing Category_Mapping_Reference is not a list; "
                        "ignoring and starting from empty.",
                    )
                    print(
                        "DEBUG: STEP E – existing reference not a list; "
                        "starting from empty"
                    )
                    old_reference = []
            except ClientError as ce_ref:
                log_warning(
                    logger,
                    "STEP E: ClientError while loading existing reference; "
                    "starting from empty.",
                )
                log_warning(logger, repr(ce_ref))
                print(
                    "DEBUG: STEP E – ClientError while loading reference; "
                    "starting from empty"
                )
                old_reference = []
            except Exception as e_ref:
                log_warning(
                    logger,
                    "STEP E: Unexpected error while loading existing reference; "
                    "starting from empty.",
                )
                log_warning(logger, repr(e_ref))
                print(
                    "DEBUG: STEP E – error while loading reference; "
                    "starting from empty"
                )
                old_reference = []
        else:
            log_info(
                logger,
                "STEP E: No existing Category_Mapping_Reference found; "
                "new one will be created.",
            )
            print("DEBUG: STEP E – no existing Category_Mapping_Reference found")

        # Build mapping pim_category_id -> existing category object
        ref_by_cat = {}
        for cat in old_reference:
            if not isinstance(cat, dict):
                continue
            cid = cat.get("pim_category_id")
            cid_str = str(cid).strip() if cid is not None else None
            if not cid_str:
                continue
            if cid_str not in ref_by_cat:
                ref_by_cat[cid_str] = cat

        # Start new reference by copying categories and clearing mapping_methods
        new_ref_by_cat = {}
        for cid_str, cat in ref_by_cat.items():
            new_cat = dict(cat)
            # Ensure vendor_mappings is a list
            vm = new_cat.get("vendor_mappings")
            if vm is None:
                vm = []
            new_cat["vendor_mappings"] = vm
            # Clear mapping_methods; will rebuild from rule_proposals
            new_cat["mapping_methods"] = []
            new_ref_by_cat[cid_str] = new_cat

        # Build rules_by_cat from rule_proposals, using accepted vendor_status
        accepted_statuses = {"new", "supported", "not_impacted"}
        rules_by_cat = {}
        total_rule_count = len(rule_proposals)
        accepted_rule_count = 0
        skipped_status_count = 0
        skipped_missing_category_count = 0
        skipped_missing_values_count = 0

        for rule in rule_proposals:
            stats = rule.get("stats") or {}
            vendor_status = stats.get("vendor_status")
            if vendor_status not in accepted_statuses:
                # skip violated or unknown
                skipped_status_count += 1
                continue

            pim_category_id = rule.get("pim_category_id")
            if pim_category_id is None:
                skipped_missing_category_count += 1
                continue
            cid_str = str(pim_category_id).strip()
            if not cid_str:
                skipped_missing_category_count += 1
                continue

            field_name = rule.get("field_name", "KEYWORD")
            operator = rule.get("operator", "contains_any")
            raw_values_include = rule.get("values_include") or []
            values_include = []
            for v in raw_values_include:
                if v is None:
                    continue
                v_str = str(v).strip()
                if v_str:
                    values_include.append(v_str)
            if not values_include:
                skipped_missing_values_count += 1
                continue

            raw_values_exclude = rule.get("values_exclude") or []
            values_exclude = []
            for v in raw_values_exclude:
                if v is None:
                    continue
                v_str = str(v).strip()
                if v_str:
                    values_exclude.append(v_str)

            method_obj = {
                "field_name": field_name,
                "values_include": values_include,
                "values_exclude": values_exclude,
                "operator": operator,
            }

            rules_by_cat.setdefault(cid_str, []).append(method_obj)
            accepted_rule_count += 1

        log_info(
            logger,
            "STEP E: Rule proposal filtering summary - "
            f"total={total_rule_count}, accepted={accepted_rule_count}, "
            f"skipped_by_status={skipped_status_count}, "
            f"skipped_missing_category={skipped_missing_category_count}, "
            f"skipped_missing_values={skipped_missing_values_count}",
        )
        print(
            "DEBUG: STEP E – rule proposal filtering summary: "
            f"total={total_rule_count}, accepted={accepted_rule_count}, "
            f"skipped_by_status={skipped_status_count}, "
            f"skipped_missing_category={skipped_missing_category_count}, "
            f"skipped_missing_values={skipped_missing_values_count}"
        )

        # Ensure new categories from rules exist in new_ref_by_cat
        for cid_str, methods in rules_by_cat.items():
            if cid_str not in new_ref_by_cat:
                new_cat = {
                    "pim_category_id": cid_str,
                    "pim_category_name": category_name_map.get(cid_str),
                    "pim_category_path": category_path_map.get(cid_str),
                    "vendor_mappings": [],
                    "mapping_methods": [],
                }
                new_ref_by_cat[cid_str] = new_cat

        # Attach mapping_methods per category, deduplicated
        def rule_signature(method: dict):
            try:
                field_name = method.get("field_name")
                operator = method.get("operator")
                vi = tuple(sorted(str(v) for v in (method.get("values_include") or [])))
                ve = tuple(sorted(str(v) for v in (method.get("values_exclude") or [])))
                return (field_name, operator, vi, ve)
            except Exception:
                return None

        for cid_str, methods in rules_by_cat.items():
            cat = new_ref_by_cat.get(cid_str)
            if cat is None:
                continue
            mm_list = cat.get("mapping_methods") or []
            sig_set = set()
            for m in mm_list:
                sig = rule_signature(m)
                if sig is not None:
                    sig_set.add(sig)
            for m in methods:
                sig = rule_signature(m)
                if sig is None:
                    continue
                if sig in sig_set:
                    continue
                mm_list.append(m)
                sig_set.add(sig)
            cat["mapping_methods"] = mm_list
            new_ref_by_cat[cid_str] = cat

        # Build old_rules_by_cat and new_rules_by_cat for change log
        old_rules_by_cat = {}
        for cid_str, cat in ref_by_cat.items():
            methods = cat.get("mapping_methods") or []
            rule_map = {}
            for m in methods:
                sig = rule_signature(m)
                if sig is None:
                    continue
                rule_map[sig] = m
            old_rules_by_cat[cid_str] = rule_map

        new_rules_by_cat = {}
        for cid_str, cat in new_ref_by_cat.items():
            methods = cat.get("mapping_methods") or []
            rule_map = {}
            for m in methods:
                sig = rule_signature(m)
                if sig is None:
                    continue
                rule_map[sig] = m
            new_rules_by_cat[cid_str] = rule_map

        # Prepare final reference list
        new_reference_list = list(new_ref_by_cat.values())
        try:
            new_reference_list.sort(
                key=lambda c: str(c.get("pim_category_id") or "")
            )
        except Exception as e_sort:
            log_warning(
                logger,
                f"STEP E: Could not sort new reference list by pim_category_id: {repr(e_sort)}",
            )
            print(
                "DEBUG: STEP E – sorting new reference list failed; "
                "continuing unsorted"
            )

        # Compute record count robustly
        canonical_mappings_prefix = "canonical_mappings"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        # Determine record count based on object type
        try:
            # Check if it's a Spark DataFrame
            if hasattr(new_reference_list, 'count') and callable(getattr(new_reference_list, 'count')):
                record_count = new_reference_list.count()
            # Check if it's a pandas DataFrame
            elif hasattr(new_reference_list, 'iloc'):
                record_count = len(new_reference_list)
            # Check if it's a list
            elif isinstance(new_reference_list, list):
                record_count = len(new_reference_list)
            # Otherwise try len()
            else:
                try:
                    record_count = len(new_reference_list)
                except (TypeError, AttributeError):
                    record_count = 1
        except Exception:
            record_count = 1
        
        log_info(logger, f"Category_Mapping_Reference record_count={record_count}")
        
        if record_count == 0:
            log_error(logger, "Category_Mapping_Reference has 0 records; cannot write empty reference file")
            raise RuntimeError("Category_Mapping_Reference has 0 records")
        
        # New reference key (timestamped only, no stable version)
        new_ref_key = os.path.join(canonical_mappings_prefix, f"Category_Mapping_Reference_{timestamp}.json")

        # Write new reference file
        current_step = "STEP E: Write new Category_Mapping_Reference"
        log_info(logger, current_step)
        body_ref = json.dumps(new_reference_list, indent=2, ensure_ascii=False)
        log_info(
            logger,
            f"STEP E: Writing new Category_Mapping_Reference to "
            f"s3://{input_bucket}/{new_ref_key}",
        )
        print(
            "DEBUG: STEP E – writing new Category_Mapping_Reference to: "
            f"s3://{input_bucket}/{new_ref_key}"
        )
        s3_client.put_object(
            Bucket=input_bucket,
            Key=new_ref_key,
            Body=body_ref.encode("utf-8"),
        )

        # Compute change log (added / removed rules per category)
        current_step = "STEP E: Compute and write rule change log"
        log_info(logger, current_step)

        changes = []
        all_cids = set(old_rules_by_cat.keys()) | set(new_rules_by_cat.keys())
        for cid_str in all_cids:
            old_map = old_rules_by_cat.get(cid_str, {})
            new_map = new_rules_by_cat.get(cid_str, {})
            old_sigs = set(old_map.keys())
            new_sigs = set(new_map.keys())
            added_sigs = new_sigs - old_sigs
            removed_sigs = old_sigs - new_sigs
            if not added_sigs and not removed_sigs:
                continue
            cat_obj = new_ref_by_cat.get(cid_str) or ref_by_cat.get(cid_str) or {}
            change_entry = {
                "pim_category_id": cid_str,
                "pim_category_name": cat_obj.get("pim_category_name"),
                "added_rules": [new_map[s] for s in added_sigs],
                "removed_rules": [old_map[s] for s in removed_sigs],
            }
            changes.append(change_entry)

        change_log = {
            "vendor_name": vendor_name,
            "timestamp": timestamp,
            "old_reference_key": latest_ref_key,
            "new_reference_key": new_ref_key,
            "changes": changes,
        }

        change_key = (
            "canonical_mappings/"
            f"Category_Mapping_RuleChanges_{vendor_name}_{timestamp}.json"
        )

        body_change = json.dumps(change_log, indent=2, ensure_ascii=False)
        log_info(
            logger,
            f"STEP E: Writing rule change log to s3://{input_bucket}/{change_key}",
        )
        print(
            "DEBUG: STEP E – writing rule change log to: "
            f"s3://{input_bucket}/{change_key}"
        )
        s3_client.put_object(
            Bucket=input_bucket,
            Key=change_key,
            Body=body_change.encode("utf-8"),
        )

        # =========================================================
        # STEP E: Write Category_Mapping_StableTrainingDataset.json if not exists
        # =========================================================
        current_step = "STEP E: Write Category_Mapping_StableTrainingDataset.json if not exists"
        stable_training_dataset_key = os.path.join(
            canonical_mappings_prefix,
            "Category_Mapping_StableTrainingDataset.json"
        )
        
        # Check if the stable training dataset already exists
        stable_dataset_exists = False
        try:
            s3_client.head_object(Bucket=input_bucket, Key=stable_training_dataset_key)
            stable_dataset_exists = True
            log_info(
                logger,
                f"Stable training dataset already exists at s3://{input_bucket}/{stable_training_dataset_key}"
            )
            print(
                f"DEBUG: STEP E – stable training dataset already exists at: "
                f"s3://{input_bucket}/{stable_training_dataset_key}"
            )
        except ClientError as ce:
            # If the error is 404 (Not Found), the object doesn't exist
            error_code = ce.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                stable_dataset_exists = False
                log_info(
                    logger,
                    f"Stable training dataset does not exist at s3://{input_bucket}/{stable_training_dataset_key}; will create it"
                )
                print(
                    f"DEBUG: STEP E – stable training dataset does not exist; will create it"
                )
            else:
                # Other errors should be logged but not fail the job
                log_warning(
                    logger,
                    f"Error checking existence of stable training dataset: {repr(ce)}"
                )
                print(
                    f"DEBUG: STEP E – error checking stable dataset existence: {repr(ce)}"
                )
        except Exception as e_head:
            # Other exceptions should be logged but not fail the job
            log_warning(
                logger,
                f"Unexpected error checking existence of stable training dataset: {repr(e_head)}"
            )
            print(
                f"DEBUG: STEP E – unexpected error checking stable dataset existence: {repr(e_head)}"
            )
        
        # Write the stable training dataset if it doesn't exist
        if not stable_dataset_exists:
            try:
                stable_dataset_body = json.dumps(final_records, indent=2, ensure_ascii=False)
                s3_client.put_object(
                    Bucket=input_bucket,
                    Key=stable_training_dataset_key,
                    Body=stable_dataset_body.encode("utf-8"),
                )
                log_info(
                    logger,
                    f"STEP E: Stable training dataset written to s3://{input_bucket}/{stable_training_dataset_key}"
                )
                print(
                    f"DEBUG: STEP E – stable training dataset written to: "
                    f"s3://{input_bucket}/{stable_training_dataset_key}"
                )
            except Exception as e_write:
                log_warning(
                    logger,
                    f"Error writing stable training dataset: {repr(e_write)}"
                )
                print(
                    f"DEBUG: STEP E – error writing stable training dataset: {repr(e_write)}"
                )

        log_info(logger, "========== JOB END (SUCCESS) ==========")
        print("DEBUG: JOB END – success path reached")
        job.commit()
        return

    except Exception as e:
        log_error(
            logger,
            f"Job failed in logical step '{current_step}' with exception.",
        )
        log_error(logger, repr(e))
        log_error(logger, traceback.format_exc())
        print("FATAL: Job failed with exception in step:", current_step, repr(e))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
