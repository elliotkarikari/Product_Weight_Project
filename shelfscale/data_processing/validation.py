import pandas as pd
import logging
from typing import Dict, List, Any
import numpy as np 

logger = logging.getLogger(__name__)

# Mapping from schema types to pandas/numpy dtypes
# This mapping helps bridge schema definitions with pandas' specific dtypes.
TYPE_MAPPING = {
    'str': ['object', 'string'], 
    'float': ['float64', 'float32', 'float16'],
    'int': ['int64', 'int32', 'int16', 'int8', 'Int64'], 
    'bool': ['bool']
}


def validate_schema(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Validates a DataFrame against a predefined schema.

    Args:
        df: The pandas DataFrame to validate.
        schema: A dictionary defining the schema. 
                Example: {'col_name': {'type': 'str', 'required': True, 'nullable': False}}

    Returns:
        A list of error messages. An empty list means validation passed.
    """
    errors: List[str] = []

    if df is None:
        errors.append("DataFrame is None. Cannot perform validation.")
        logger.error("DataFrame is None in validate_schema.")
        return errors
        
    if not isinstance(df, pd.DataFrame):
        errors.append(f"Expected a pandas DataFrame, but got {type(df)}.")
        logger.error(f"Invalid input type to validate_schema: {type(df)}.")
        return errors

    # 1. Check for required columns
    for col_name, col_props in schema.items():
        if col_props.get('required', False) and col_name not in df.columns:
            msg = f"Required column '{col_name}' is missing."
            errors.append(msg)
            logger.warning(msg)

    # 2. Check data types and nullable properties for existing columns
    for col_name in df.columns:
        if col_name in schema:
            col_props = schema[col_name]
            actual_dtype = str(df[col_name].dtype)
            
            # Check type
            expected_type_str = col_props.get('type')
            if expected_type_str:
                expected_pandas_dtypes = TYPE_MAPPING.get(expected_type_str)
                if expected_pandas_dtypes:
                    if actual_dtype not in expected_pandas_dtypes:
                        # Special handling for 'int' column that pandas might interpret as 'float64' if it contains all NaNs
                        # and was not explicitly cast to a nullable integer type like 'Int64'.
                        if expected_type_str == 'int' and actual_dtype == 'float64' and df[col_name].isnull().all() and col_props.get('nullable', True):
                            logger.info(f"Column '{col_name}' (dtype: {actual_dtype}) contains all NaNs. Considered valid for schema type 'int' with nullable:True.")
                        # Special handling for 'object' dtype columns that are expected to be 'int' or 'float' but might contain mixed types or uncast NaNs.
                        elif expected_type_str == 'int' and actual_dtype == 'object':
                             try: 
                                 numeric_series = pd.to_numeric(df[col_name], errors='raise')
                                 if numeric_series.dropna().apply(lambda x: float(x).is_integer()).all(): # Check if all numbers are integers
                                     logger.info(f"Column '{col_name}' (dtype: {actual_dtype}) contains numbers that are all integers (or NaN). Considered valid for schema type 'int'.")
                                 else:
                                     raise ValueError("Not all numeric values are integers.")
                             except Exception:
                                msg = (f"Column '{col_name}': Expected type(s) '{expected_pandas_dtypes}' "
                                       f"(schema type '{expected_type_str}'), but found '{actual_dtype}'. Contains non-integer or non-convertible values.")
                                errors.append(msg)
                                logger.warning(msg)
                        elif expected_type_str == 'float' and actual_dtype == 'object':
                             try: 
                                 pd.to_numeric(df[col_name], errors='raise')
                                 logger.info(f"Column '{col_name}' (dtype: {actual_dtype}) contains values convertible to float. Considered valid for schema type 'float'.")
                             except Exception:
                                msg = (f"Column '{col_name}': Expected type(s) '{expected_pandas_dtypes}' "
                                       f"(schema type '{expected_type_str}'), but found '{actual_dtype}'. Contains non-float or non-convertible values.")
                                errors.append(msg)
                                logger.warning(msg)
                        else:
                            msg = (f"Column '{col_name}': Expected type(s) '{expected_pandas_dtypes}' "
                                   f"(schema type '{expected_type_str}'), but found '{actual_dtype}'.")
                            errors.append(msg)
                            logger.warning(msg)
                else:
                    msg = f"Column '{col_name}': Unknown type '{expected_type_str}' in schema definition."
                    errors.append(msg)
                    logger.warning(msg)

            # Check nullable
            is_nullable = col_props.get('nullable', True) 
            if not is_nullable:
                if df[col_name].isnull().any():
                    already_type_error = any(f"Column '{col_name}'" in err and "type" in err for err in errors)
                    if not already_type_error:
                        msg = f"Column '{col_name}' (type: {actual_dtype}) is not nullable (nullable: False in schema) but contains NaN values."
                        errors.append(msg)
                        logger.warning(msg)
                        
    if not errors:
        logger.info(f"Schema validation passed for DataFrame (checked columns defined in schema: {list(schema.keys())}).")
    else:
        logger.warning(f"Schema validation failed with {len(errors)} error(s).")
        
    return errors

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) 

    sample_schema = {
        'ID': {'type': 'int', 'required': True, 'nullable': False},
        'Name': {'type': 'str', 'required': True, 'nullable': False},
        'Value': {'type': 'float', 'required': False, 'nullable': True},
        'IsActive': {'type': 'bool', 'required': False, 'nullable': False},
        'OptionalIntAllNaN': {'type': 'int', 'required': False, 'nullable': True},
        'OptionalFloatAllNaN': {'type': 'float', 'required': False, 'nullable': True},
        'IntAsObject': {'type': 'int', 'required': False, 'nullable': True},
        'FloatAsObject': {'type': 'float', 'required': False, 'nullable': True}
    }

    data_pass = pd.DataFrame({
        'ID': pd.Series([1, 2, 3], dtype='int64'),
        'Name': pd.Series(['Alice', 'Bob', 'Charlie'], dtype='string'), 
        'Value': pd.Series([10.5, np.nan, 20.0], dtype='float64'),
        'IsActive': pd.Series([True, False, True], dtype='bool'),
        'OptionalIntAllNaN': pd.Series([None, None, None], dtype=pd.Int64Dtype()),
        'OptionalFloatAllNaN': pd.Series([np.nan, np.nan, np.nan], dtype='float64'),
        'IntAsObject': pd.Series([1, 2, None, 3, '4'], dtype='object'), 
        'FloatAsObject': pd.Series([1.0, '2.5', None, 3.0, '3.14'], dtype='object')
    })
    
    print("--- Validating data_pass ---")
    errors_pass = validate_schema(data_pass, sample_schema)
    print(f"Validation Errors (data_pass): {errors_pass}\n")

    data_fail_missing_col = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Value': [10.5, 20.0]})
    print("--- Validating data_fail_missing_col ---")
    errors_missing_col = validate_schema(data_fail_missing_col, sample_schema)
    print(f"Validation Errors (data_fail_missing_col): {errors_missing_col}\n")

    data_fail_type_id = pd.DataFrame({'ID': [1, 2, '3A'], 'Name': ['Alice', 'Bob', 'Charlie']}) 
    print("--- Validating data_fail_type_id (ID has '3A') ---")
    errors_type_id = validate_schema(data_fail_type_id, sample_schema)
    print(f"Validation Errors (data_fail_type_id): {errors_type_id}\n")
    
    data_fail_type_isactive = pd.DataFrame({'ID': [1], 'Name': ['A'], 'IsActive': ['True', False, None]})
    print("--- Validating data_fail_type_isactive (IsActive has string 'True' and None for non-nullable) ---")
    errors_type_isactive = validate_schema(data_fail_type_isactive, sample_schema)
    print(f"Validation Errors (data_fail_type_isactive): {errors_type_isactive}\n")

    data_fail_nullable_id = pd.DataFrame({'ID': pd.Series([1, 2, None], dtype=pd.Int64Dtype()), 'Name': ['Alice', 'Bob', 'Charlie']})
    print("--- Validating data_fail_nullable_id (ID is not nullable but contains pd.NA) ---")
    errors_nullable_id = validate_schema(data_fail_nullable_id, sample_schema)
    print(f"Validation Errors (data_fail_nullable_id): {errors_nullable_id}\n")

    data_fail_nullable_isactive = pd.DataFrame({'ID': [1], 'Name': ['Test'], 'IsActive': pd.Series([True, pd.NA, False], dtype='boolean')}) 
    print("--- Validating data_fail_nullable_isactive (IsActive is not nullable but contains pd.NA) ---")
    errors_nullable_isactive = validate_schema(data_fail_nullable_isactive, sample_schema)
    print(f"Validation Errors (data_fail_nullable_isactive): {errors_nullable_isactive}\n")

    data_empty_cols_no_rows = pd.DataFrame(columns=sample_schema.keys())
    print("--- Validating data_empty_cols_no_rows ---")
    errors_empty_rows = validate_schema(data_empty_cols_no_rows, sample_schema)
    print(f"Validation Errors (data_empty_cols_no_rows): {errors_empty_rows}\n")

    data_empty_no_cols = pd.DataFrame()
    print("--- Validating data_empty_no_cols ---")
    errors_empty_no_cols = validate_schema(data_empty_no_cols, sample_schema)
    print(f"Validation Errors (data_empty_no_cols): {errors_empty_no_cols}\n")

    data_none_df = None
    print("--- Validating None DataFrame ---")
    errors_none = validate_schema(data_none_df, sample_schema)
    print(f"Validation Errors (data_none_df): {errors_none}\n")

    data_optional_int_all_nan_as_float = pd.DataFrame({
        'ID': [1,2,3], 'Name': ['A','B','C'],
        'OptionalIntAllNaN': pd.Series([np.nan, np.nan, np.nan], dtype='float64')
    })
    print("--- Validating data_optional_int_all_nan_as_float ---")
    errors_int_all_nan_float = validate_schema(data_optional_int_all_nan_as_float, sample_schema)
    print(f"Validation Errors (data_optional_int_all_nan_as_float): {errors_int_all_nan_float}\n") 
    
    data_int_as_object_invalid_float = pd.DataFrame({
        'ID': [1,2,3], 'Name': ['A','B','C'],
        'IntAsObject': pd.Series([10, '20.5', None], dtype='object') 
    })
    print("--- Validating data_int_as_object_invalid_float (IntAsObject contains '20.5') ---")
    errors_int_as_object_invalid = validate_schema(data_int_as_object_invalid_float, sample_schema)
    print(f"Validation Errors (data_int_as_object_invalid_float): {errors_int_as_object_invalid}\n")

    data_float_as_object_invalid_str = pd.DataFrame({
        'ID': [1,2,3], 'Name': ['A','B','C'],
        'FloatAsObject': pd.Series([1.0, '2.5X', None], dtype='object') 
    })
    print("--- Validating data_float_as_object_invalid_str (FloatAsObject contains '2.5X') ---")
    errors_float_as_object_invalid = validate_schema(data_float_as_object_invalid_str, sample_schema)
    print(f"Validation Errors (data_float_as_object_invalid_str): {errors_float_as_object_invalid}\n")
