from .argparse import InseqArgumentParser
from .cache import INSEQ_ARTIFACTS_CACHE, INSEQ_HOME_CACHE, cache_results
from .errors import (
    InseqDeprecationWarning,
    LengthMismatchError,
    MissingAttributionMethodError,
    UnknownAttributionMethodError,
)
from .import_utils import (
    is_captum_available,
    is_datasets_available,
    is_ipywidgets_available,
    is_joblib_available,
    is_scikitlearn_available,
    is_sentencepiece_available,
    is_transformers_available,
)
from .misc import (
    aggregate_token_pair,
    aggregate_token_sequence,
    bin_str_to_ndarray,
    drop_padding,
    extract_signature_args,
    find_char_indexes,
    format_input_texts,
    get_cls_from_instance_type,
    get_module_name_from_object,
    gzip_compress,
    gzip_decompress,
    hashodict,
    identity_fn,
    isnotebook,
    lists_of_numbers_to_ndarray,
    ndarray_to_bin_str,
    optional,
    pad,
    pretty_dict,
    pretty_list,
    pretty_tensor,
    rgetattr,
    save_to_file,
    scalar_to_numpy,
)
from .registry import Registry, get_available_methods
from .serialization import json_advanced_dump, json_advanced_dumps, json_advanced_load, json_advanced_loads
from .torch_utils import (
    abs_max,
    aggregate_contiguous,
    check_device,
    euclidean_distance,
    get_default_device,
    get_front_padding,
    get_sequences_from_batched_steps,
    normalize_attributions,
    prod_fn,
    remap_from_filtered,
    sum_fn,
    sum_normalize_attributions,
)

__all__ = [
    "LengthMismatchError",
    "MissingAttributionMethodError",
    "UnknownAttributionMethodError",
    "cache_results",
    "optional",
    "identity_fn",
    "pad",
    "pretty_list",
    "pretty_tensor",
    "pretty_dict",
    "aggregate_token_pair",
    "aggregate_token_sequence",
    "format_input_texts",
    "rgetattr",
    "get_available_methods",
    "isnotebook",
    "find_char_indexes",
    "extract_signature_args",
    "remap_from_filtered",
    "drop_padding",
    "normalize_attributions",
    "sum_normalize_attributions",
    "aggregate_contiguous",
    "abs_max",
    "prod_fn",
    "sum_fn",
    "get_front_padding",
    "get_sequences_from_batched_steps",
    "euclidean_distance",
    "Registry",
    "INSEQ_HOME_CACHE",
    "INSEQ_ARTIFACTS_CACHE",
    "InseqArgumentParser",
    "is_ipywidgets_available",
    "is_scikitlearn_available",
    "is_transformers_available",
    "is_sentencepiece_available",
    "is_datasets_available",
    "is_captum_available",
    "is_joblib_available",
    "check_device",
    "get_default_device",
    "ndarray_to_bin_str",
    "hashodict",
    "InseqDeprecationWarning",
    "get_module_name_from_object",
    "gzip_compress",
    "gzip_decompress",
    "save_to_file",
    "json_advanced_dump",
    "json_advanced_dumps",
    "bin_str_to_ndarray",
    "lists_of_numbers_to_ndarray",
    "scalar_to_numpy",
    "get_cls_from_instance_type",
    "json_advanced_loads",
    "json_advanced_load",
]
