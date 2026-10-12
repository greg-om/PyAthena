# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import csv
import io
from collections import defaultdict
from unittest.mock import ANY, MagicMock, PropertyMock, patch, sentinel

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem
from pandas.testing import assert_frame_equal

from pyathena.error import OperationalError
from pyathena.pandas.converter import DefaultPandasTypeConverter
from pyathena.pandas.result_set import (
    AthenaPandasResultSet,
    PandasDataFrameIterator,
    _no_trunc_date,
    _read_csv_with_pyarrow,
)


class TestPandasDataFrameIterator:
    @pytest.mark.parametrize(
        ("csv", "read_csv_kwargs"),
        [
            ("id,kind\n10,a\n11,a\n12,b\n13,b\n14,c\n", {}),
            ("id,kind\n10,a\n11,a\n12,b\n13,b\n14,c\n", {"index_col": "id"}),
            ("id,kind\n10,c\n11,c\n12,a\n13,a\n14,b\n", {"dtype": {"kind": "category"}}),
            ("id,kind\n10,\n11,\n12,b\n13,a\n14,\n", {"dtype": {"kind": "category"}}),
            (
                "id,kind\n10,c\n11,c\n12,a\n13,a\n14,b\n",
                {"dtype": {"kind": pd.CategoricalDtype(ordered=True)}},
            ),
            (
                "id,kind\n10,c\n11,c\n12,a\n13,a\n14,b\n",
                {"index_col": "kind", "dtype": {"kind": "category"}},
            ),
        ],
        ids=[
            "default_index",
            "index_col",
            "category",
            "category_null_chunk",
            "ordered_category",
            "category_index",
        ],
    )
    def test_as_pandas_matches_whole_read(self, csv, read_csv_kwargs):
        """Joining the chunks gives the DataFrame that reading the whole file gives."""
        expected = pd.read_csv(io.StringIO(csv), **read_csv_kwargs)
        reader = pd.read_csv(io.StringIO(csv), chunksize=2, **read_csv_kwargs)
        df_iter = PandasDataFrameIterator(reader, _no_trunc_date)

        pd.testing.assert_frame_equal(df_iter.as_pandas(), expected)

    def test_as_pandas_remaining_chunks(self):
        """After a chunk was read, the remaining chunks keep their row numbers."""
        reader = pd.read_csv(io.StringIO("n\n1\n2\n3\n4\n5\n"), chunksize=2)
        df_iter = PandasDataFrameIterator(reader, _no_trunc_date)
        next(df_iter)

        df = df_iter.as_pandas()
        assert df["n"].tolist() == [3, 4, 5]
        assert df.index.tolist() == [2, 3, 4]
        assert df_iter.as_pandas().empty

    def test_as_pandas_single_dataframe(self):
        """A DataFrame that was read at once is returned as is."""
        df = pd.DataFrame({"n": [1, 2]})
        df_iter = PandasDataFrameIterator(df, _no_trunc_date)

        assert df_iter.as_pandas() is df


_FS = sentinel.pyathena_fs
_USER_FS = sentinel.user_fs


class TestAthenaPandasResultSet:
    @pytest.mark.parametrize("engine", ["auto", "c", "python", "pyarrow"])
    @pytest.mark.parametrize(
        ("output_location", "file_size_bytes", "pyarrow_engine"),
        [
            ("s3://bucket/result.txt", None, "c"),
            ("s3://bucket/result.txt", 99, "c"),
            ("s3://bucket/result.txt", 100, "c"),
            ("s3://bucket/result.txt", 101, "c"),
            ("s3://bucket/result.csv", None, "pyarrow"),
            ("s3://bucket/result.csv", 99, "c"),
            ("s3://bucket/result.csv", 100, "pyarrow"),
            ("s3://bucket/result.csv", 101, "pyarrow"),
            (None, None, "pyarrow"),
        ],
    )
    def test_get_csv_engine_result_format(
        self, engine, output_location, file_size_bytes, pyarrow_engine
    ):
        """PyArrow falls back for DDL text while C and Python choices are preserved."""
        result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
        result_set._query_execution = MagicMock(output_location=output_location)
        result_set._metadata = None
        result_set._converter = DefaultPandasTypeConverter()
        result_set._engine = engine
        result_set._chunksize = None
        result_set._quoting = csv.QUOTE_ALL
        result_set._kwargs = {}

        expected = {"auto": "c", "c": "c", "python": "python", "pyarrow": pyarrow_engine}[engine]
        assert result_set._get_csv_engine(file_size_bytes) == expected

    @pytest.mark.parametrize("infer_string", [True, False])
    @pytest.mark.parametrize("read_csv_kwargs", [{}, {"keep_default_na": True}])
    @pytest.mark.parametrize(
        ("data", "names", "expected_names"),
        [
            ("001\n007\n1e3\n" * 10, ["tab_name"], ["001", "007", "1e3"] * 10),
            (
                "001                 \n007                 \n1e3                 \n" * 10,
                ["field"],
                ["001                 ", "007                 ", "1e3                 "] * 10,
            ),
            (
                (
                    "001                 \tint                 \t\n"
                    "007                 \tint                 \t\n"
                    "1e3                 \tint                 \t\n"
                )
                * 10,
                ["col_name", "data_type", "comment"],
                ["001                 ", "007                 ", "1e3                 "] * 10,
            ),
        ],
        ids=["show_tables", "show_columns", "describe"],
    )
    def test_read_csv_ddl_preserves_numeric_looking_names(
        self, data, names, expected_names, read_csv_kwargs, infer_string
    ):
        """Reading DDL results preserves numeric-looking names and their padding."""
        data = data.encode()
        assert len(data) >= AthenaPandasResultSet.PYARROW_MIN_FILE_SIZE_BYTES

        result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
        result_set._query_execution = MagicMock(
            output_location="s3://bucket/result.txt", substatement_type=None
        )
        result_set._converter = DefaultPandasTypeConverter()
        result_set._engine = "pyarrow"
        result_set._chunksize = None
        result_set._auto_optimize_chunksize = False
        result_set._quoting = csv.QUOTE_ALL
        result_set._keep_default_na = False
        result_set._na_values = ("",)
        result_set._kwargs = read_csv_kwargs
        result_set._block_size = None
        result_set._cache_type = None
        result_set._fs = MagicMock()
        result_set._fs.open.return_value = stream = io.BytesIO(data)
        description = [(name, "string", None, None, 0, 0, "UNKNOWN") for name in names]

        with (
            pd.option_context("future.infer_string", infer_string),
            patch.object(
                AthenaPandasResultSet,
                "description",
                new_callable=PropertyMock,
                return_value=description,
            ),
            patch.object(result_set, "_get_content_length", return_value=len(data)),
            patch("pandas.read_csv", wraps=pd.read_csv) as read_csv,
        ):
            expected = pd.read_csv(io.BytesIO(data), **result_set._get_csv_read_options("c", None))
            read_csv.reset_mock()
            actual = result_set._read_csv()

        assert_frame_equal(actual, expected, check_exact=True)
        assert actual.iloc[:, 0].tolist() == expected_names
        read_csv.assert_called_once()
        assert read_csv.call_args.kwargs["engine"] == "c"
        result_set._fs.open.assert_called_once_with(
            "s3://bucket/result.txt", mode="rb", **result_set._get_csv_open_options(True)
        )
        assert stream.closed

    @pytest.mark.parametrize(
        ("chunksize", "execute_kwargs", "expected"),
        [
            (None, {}, False),
            (10, {}, True),
            (10, {"columns": ["a", "b"]}, True),
            (10, {"columns": ("a",)}, True),
            (10, {"dtype_backend": "numpy_nullable"}, True),
            (10, {"dtype_backend": "pyarrow", "columns": ["a"]}, True),
            (10, {"columns": []}, False),
            (10, {"columns": ["a", "a"]}, False),
            (10, {"columns": "a"}, False),
            (10, {"columns": [0]}, False),
            (10, {"columns": ["s.a"]}, False),
            (10, {"dtype_backend": None}, False),
            (10, {"dtype_backend": "invalid"}, False),
            (10, {"filters": [("a", "=", 1)]}, False),
            (10, {"storage_options": {}}, False),
        ],
    )
    def test_reads_parquet_in_chunks(self, chunksize, execute_kwargs, expected):
        """Only chunksize with supported read options reads UNLOAD results in chunks."""
        result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
        result_set._chunksize = chunksize
        result_set._kwargs = execute_kwargs
        assert result_set._reads_parquet_in_chunks() is expected

    @pytest.mark.parametrize(
        ("block_size", "cache_type", "prefetch", "expected"),
        [
            (
                None,
                None,
                True,
                {
                    "block_size": 32 * 2**20,
                    "cache_type": "background",
                    "cache_options": {"maxblocks": 2},
                },
            ),
            (None, None, False, {}),
            (5 * 2**20, None, True, {}),
            (None, "bytes", True, {}),
            (5 * 2**20, "readahead", True, {}),
        ],
    )
    def test_get_csv_open_options(self, block_size, cache_type, prefetch, expected):
        """Whole CSV results are prefetched unless the cursor sets block_size or cache_type."""
        result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
        result_set._block_size = block_size
        result_set._cache_type = cache_type
        assert result_set._get_csv_open_options(prefetch) == expected

    @pytest.mark.parametrize(
        ("execute_kwargs", "path", "filesystem_kwargs"),
        [
            ({}, "bucket/unload/", {"filesystem": _FS}),
            ({"filesystem": _USER_FS}, "bucket/unload/", {"filesystem": _USER_FS}),
            ({"filesystem": None}, "s3://bucket/unload/", {"filesystem": None}),
            ({"storage_options": {"anon": True}}, "s3://bucket/unload/", {}),
            ({"storage_options": None}, "s3://bucket/unload/", {}),
        ],
        ids=["default", "filesystem", "filesystem-none", "storage-options", "storage-options-none"],
    )
    def test_read_parquet_filesystem(self, execute_kwargs, path, filesystem_kwargs):
        """filesystem or storage_options given to execute() replace PyAthena's filesystem.

        No AWS calls; the manifest and pandas.read_parquet are mocked.
        """
        result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)  # bypass __init__
        result_set._unload_location = None
        result_set._engine = "pyarrow"
        result_set._fs = _FS
        result_set._kwargs = dict(execute_kwargs)
        with (
            patch.object(
                AthenaPandasResultSet,
                "_read_data_manifest",
                return_value=["s3://bucket/unload/0.parquet"],
            ),
            patch("pandas.read_parquet") as read_parquet,
        ):
            result_set._read_parquet("pyarrow")
        assert read_parquet.call_args.args == (path,)
        assert read_parquet.call_args.kwargs == {
            "engine": "pyarrow",
            "use_threads": True,
            **execute_kwargs,
            **filesystem_kwargs,
        }

    def test_iter_parquet_chunks_null_in_later_file(self, tmp_path):
        """Top-level integer and boolean columns with NULL anywhere convert alike in every chunk."""
        chunks, _ = _read_unload_result(
            tmp_path,
            [
                pa.table({"i": pa.array([1, 2], pa.int64()), "b": [True, False]}),
                pa.table({"i": pa.array([None, 4], pa.int64()), "b": [None, True]}),
            ],
            chunksize=1,
        )
        whole = pd.read_parquet(tmp_path)
        assert [len(chunk) for chunk in chunks] == [1, 1, 1, 1]
        for chunk in chunks:
            assert chunk.dtypes.tolist() == whole.dtypes.tolist()
        assert_frame_equal(pd.concat(chunks), whole, check_exact=True)

    @pytest.mark.parametrize("write_statistics", [True, False])
    def test_iter_parquet_chunks_large_integers(self, tmp_path, write_statistics):
        """Integers without NULL stay exact, with or without Parquet statistics."""
        chunks, whole = _read_unload_result(
            tmp_path,
            [pa.table({"i": pa.array([9007199254740993, 1], pa.int64())})],
            chunksize=1,
            write_statistics=write_statistics,
        )
        assert [chunk["i"].dtype for chunk in chunks] == ["int64", "int64"]
        assert [int(chunk["i"].iloc[0]) for chunk in chunks] == [9007199254740993, 1]
        assert_frame_equal(pd.concat(chunks), whole, check_exact=True)

    def test_iter_parquet_chunks_nested_values(self, tmp_path):
        """Nested values convert per chunk; with dtype_backend="pyarrow" they do not."""
        table = pa.table({"li": pa.array([[1], [None]], pa.list_(pa.int64()))})
        chunks, _ = _read_unload_result(tmp_path, [table], chunksize=1)
        assert [chunk["li"].iloc[0].dtype for chunk in chunks] == ["int64", "float64"]
        chunks, whole = _read_unload_result(
            tmp_path / "pyarrow", [table], chunksize=1, dtype_backend="pyarrow"
        )
        for chunk in chunks:
            assert chunk.dtypes.tolist() == whole.dtypes.tolist()
        assert_frame_equal(pd.concat(chunks), whole, check_exact=True)

    def test_iter_parquet_chunks_reader_error(self, tmp_path):
        """Errors from creating the chunk reader are raised as OperationalError."""
        with pytest.raises(OperationalError, match="chunksize must be positive"):
            _read_unload_result(tmp_path, [pa.table({"i": [1]})], chunksize=0)


_READ_MODES = [
    pytest.param(None, {}, True, id="dataframe"),
    pytest.param(1, {}, False, id="chunksize"),
    pytest.param(None, {"iterator": True}, False, id="iterator"),
]


@pytest.mark.parametrize(("chunksize", "kwargs", "prefetch"), _READ_MODES)
def test_read_csv_prefetches_without_reader(chunksize, kwargs, prefetch):
    """Only a CSV result read in one pandas.read_csv() call uses the background cache."""
    with patch.object(
        AthenaPandasResultSet, "_get_csv_open_options", autospec=True, return_value={}
    ) as open_options:
        result, _ = _read_csv_result(
            '"v"\n"a"\n"b"\n', {"v": "varchar"}, chunksize=chunksize, **kwargs
        )
        if not prefetch:
            result.close()
    open_options.assert_called_once_with(ANY, prefetch)


@pytest.mark.parametrize(("chunksize", "kwargs", "prefetch"), _READ_MODES)
def test_read_csv_binary_stream_prefetch(chunksize, kwargs, prefetch):
    """The text stream for binary columns is opened for the same kind of read."""
    with patch.object(
        AthenaPandasResultSet,
        "_open_binary_csv_stream",
        autospec=True,
        side_effect=lambda *args, **_: io.StringIO('"b"\n"00 ff"\n'),
    ) as open_stream:
        result, _ = _read_csv_result(
            '"b"\n"00 ff"\n', {"b": "varbinary"}, chunksize=chunksize, **kwargs
        )
        if not prefetch:
            result.close()
    assert open_stream.call_args.kwargs == {"prefetch": prefetch}


# A CSV result of Athena with each type the PyArrow engine reads, then a row of NULLs.
_TYPES_CSV = (
    '"ti","si","i","bi","r","d","c","v","ml","arr","m","rw","dt","ts","ts6","tm","iv","nul","u",'
    '"empty","na"\n'
    '"1","2","3","4","1.5","2.25","ab ","plain","multi\nline ""q"", x","[1, 2]","{k=1}",'
    '"{a=1, b=x}","2024-02-29","2024-02-29 23:59:58.123","2024-02-29 23:59:58.123456",'
    '"12:34:56.789","2 00:00:00.000",,"589f6631-9c50-4f58-a121-e2608a04fc64","","NA"\n'
    ",,,,,,,,,,,,,,,,,,,,\n"
)
_TYPES = {
    "ti": "tinyint",
    "si": "smallint",
    "i": "integer",
    "bi": "bigint",
    "r": "float",
    "d": "double",
    "c": "char",
    "v": "varchar",
    "ml": "varchar",
    "arr": "array",
    "m": "map",
    "rw": "row",
    "dt": "date",
    "ts": "timestamp",
    "ts6": "timestamp",
    "tm": "time",
    "iv": "interval day to second",
    "nul": "unknown",
    "u": "uuid",
    "empty": "varchar",
    "na": "varchar",
}


def _pyarrow_read_csv_kwargs(types, tab_separated=False, dtype_overrides=None, **kwargs):
    """Build the pandas.read_csv() options with AthenaPandasResultSet._get_csv_read_options().

    Args:
        types: The Athena types of the result columns, keyed by column name.
        tab_separated: Whether the result is a tab-separated ``.txt`` file.
        dtype_overrides: Entries to add to or replace in the default dtype mapping.
            A ``dtype`` in kwargs instead replaces the entire mapping.
        **kwargs: The pandas.read_csv() options given to ``execute()``.

    Returns:
        The options for the PyArrow engine.
    """
    result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
    result_set._converter = DefaultPandasTypeConverter()
    result_set._keep_default_na = False
    result_set._na_values = ("",)
    result_set._quoting = 1
    result_set._kwargs = kwargs
    description = [(name, type_, None, None, 0, 0, "UNKNOWN") for name, type_ in types.items()]
    location = f"s3://bucket/result.{'txt' if tab_separated else 'csv'}"
    with (
        patch.object(
            AthenaPandasResultSet,
            "description",
            new_callable=PropertyMock,
            return_value=description,
        ),
        patch.object(
            AthenaPandasResultSet,
            "output_location",
            new_callable=PropertyMock,
            return_value=location,
        ),
    ):
        if dtype_overrides is not None:
            result_set._kwargs["dtype"] = {**result_set.dtypes, **dtype_overrides}
        assert result_set._reads_csv_with_pyarrow()
        return result_set._get_csv_read_options("pyarrow", None)


def _read_unload_result(tmp_path, tables, chunksize, write_statistics=True, **kwargs):
    """Write Parquet files and read them in chunks through the result set, without AWS.

    Returns:
        The chunks, and the whole result read by pandas.read_parquet() with the same
        options.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, table in enumerate(tables):
        path = tmp_path / f"{index}.parquet"
        pq.write_table(table, path, write_statistics=write_statistics)
        paths.append(str(path))
    result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)  # bypass __init__
    result_set._fs = LocalFileSystem()
    result_set._chunksize = chunksize
    result_set._kwargs = kwargs
    result_set._unload_location = None
    result_set._parquet_null_dtypes = {}
    with patch.object(AthenaPandasResultSet, "_read_data_manifest", return_value=paths):
        assert result_set._read_parquet_footers()[0] == sum(t.num_rows for t in tables)
    chunks = list(result_set._iter_parquet_chunks())
    whole = pd.concat([pd.read_parquet(path, **kwargs) for path in paths], ignore_index=True)
    return chunks, whole


def _read_csv_result(data, types, engine="c", chunksize=None, **kwargs):
    """Read an in-memory CSV through the result set and the real pandas parser."""
    result_set = AthenaPandasResultSet.__new__(AthenaPandasResultSet)
    result_set._converter = DefaultPandasTypeConverter()
    result_set._keep_default_na = False
    result_set._na_values = ("",)
    result_set._quoting = 1
    result_set._engine = engine
    result_set._chunksize = chunksize
    result_set._auto_optimize_chunksize = False
    result_set._kwargs = kwargs
    result_set._time_columns = []
    result_set._csv_stream = None
    result_set._block_size = None
    result_set._cache_type = None
    result_set._fs = MagicMock()
    result_set._fs.open.return_value = io.BytesIO(data.encode())
    description = [(name, type_, None, None, 0, 0, "UNKNOWN") for name, type_ in types.items()]
    with (
        patch.object(
            AthenaPandasResultSet,
            "description",
            new_callable=PropertyMock,
            return_value=description,
        ),
        patch.object(
            AthenaPandasResultSet,
            "output_location",
            new_callable=PropertyMock,
            return_value="s3://bucket/result.csv",
        ),
        patch.object(
            AthenaPandasResultSet,
            "substatement_type",
            new_callable=PropertyMock,
            return_value=None,
        ),
        patch.object(AthenaPandasResultSet, "_get_content_length", return_value=len(data)),
        patch("pandas.read_csv", wraps=pd.read_csv) as read_csv,
        patch(
            "pyathena.pandas.result_set._read_csv_with_pyarrow",
            wraps=_read_csv_with_pyarrow,
        ) as read_pyarrow,
    ):
        result = result_set._read_csv()
    options = read_pyarrow.call_args.args[1] if read_pyarrow.called else read_csv.call_args.kwargs
    if isinstance(result, pd.DataFrame):
        return result_set._finish_csv_frame(result), options
    return PandasDataFrameIterator(
        result, result_set._finish_csv_frame, result_set._csv_stream
    ), options


@pytest.mark.filterwarnings("error::pandas.errors.DtypeWarning")
@pytest.mark.parametrize(
    ("json_value", "expected"), [("9007199254740993", 9007199254740993), ("true", True)]
)
@pytest.mark.parametrize(
    ("engine", "kwargs", "access"),
    [
        pytest.param("c", {}, "whole", id="c"),
        pytest.param("c", {"low_memory": True}, "whole", id="low-memory"),
        pytest.param("c", {"low_memory": False}, "whole", id="no-low-memory"),
        pytest.param("auto", {}, "whole", id="auto"),
        pytest.param("pyarrow", {}, "whole", id="pyarrow-fallback"),
        pytest.param("python", {}, "whole", id="python"),
        pytest.param("c", {"chunksize": 400005}, "iteration", id="iteration"),
        pytest.param("c", {"chunksize": 400010}, "get_chunk", id="get-chunk"),
        pytest.param("c", {"index_col": "j"}, "whole", id="index"),
        pytest.param("c", {"usecols": ["j"]}, "whole", id="usecols"),
    ],
)
def test_read_csv_json_null_without_dtype_warning(json_value, expected, engine, kwargs, access):
    """JSON NULLs in a later parser block keep exact values without a warning."""
    data = "n,j\n" + f"1,{json_value}\n" * 400000 + "2,\n" * 10
    result, options = _read_csv_result(data, {"n": "integer", "j": "json"}, engine=engine, **kwargs)
    if access != "whole":
        with result:
            first = result.get_chunk(400005) if access == "get_chunk" else next(result)
            assert len(first) == 400005
            df = pd.concat([first, *result])
    else:
        df = result
    values = df.index if kwargs.get("index_col") == "j" else df["j"]
    assert options["engine"] == ("python" if engine == "python" else "c")
    assert options.get("low_memory") == kwargs.get("low_memory")
    assert options["dtype"]["j"] is object
    assert "j" not in options["converters"]
    assert len(df) == 400010
    assert values.dtype == object
    assert type(values[0]) is type(expected)
    assert (values[:400000] == expected).all()
    assert values[-10:].tolist() == [None] * 10


@pytest.mark.filterwarnings("error::pandas.errors.DtypeWarning")
@pytest.mark.parametrize("kwargs", [{}, {"low_memory": True}])
def test_read_csv_json_numbers_in_a_parser_block_without_null(kwargs):
    """JSON integers keep their type when their parser block has a float but no NULL."""
    data = "n,j\n1,1.5\n" + "1,7\n" * 400000 + "2,\n" * 10
    df, _ = _read_csv_result(data, {"n": "integer", "j": "json"}, **kwargs)
    values = df["j"]
    assert values.dtype == object
    assert values[0] == 1.5
    assert type(values[1]) is int
    assert (values[1:400001] == 7).all()
    assert values[-10:].tolist() == [None] * 10


@pytest.mark.filterwarnings("error::pandas.errors.DtypeWarning")
@pytest.mark.parametrize("engine", ["c", "python"])
@pytest.mark.parametrize("low_memory", [None, True, False])
@pytest.mark.parametrize(
    ("text", "dtype", "values"),
    [
        pytest.param("2\n3,4", "int64", [2, 4], id="integers"),
        pytest.param("2\n3,4.5", "float64", [2.0, 4.5], id="numbers"),
        pytest.param("true\n3,false", "bool", [True, False], id="booleans"),
        pytest.param('"""a"""\n3,"""b"""', "str", ["a", "b"], id="strings"),
        pytest.param('"{""a"": 1}"\n3,[1]', object, [{"a": 1}, [1]], id="containers"),
    ],
)
def test_read_csv_json_without_null(engine, low_memory, text, dtype, values):
    """JSON without NULL gets the dtype that pandas infers for a converter's values."""
    kwargs = {} if low_memory is None else {"low_memory": low_memory}
    if engine == "python" and low_memory is not None:
        pytest.skip("low_memory is a C engine option")
    df, options = _read_csv_result(
        f"n,j\n1,{text}\n", {"n": "integer", "j": "json"}, engine=engine, **kwargs
    )
    assert options.get("low_memory") == low_memory
    assert df["j"].dtype == dtype
    assert df["j"].tolist() == values


def test_read_csv_json_multi_index_level():
    """A json column in a MultiIndex is converted."""
    df, _ = _read_csv_result(
        "n,j\n1,true\n2,false\n3,\n", {"n": "integer", "j": "json"}, index_col=["n", "j"]
    )
    level = df.index.get_level_values("j")
    assert level[:2].tolist() == [True, False]
    assert pd.isna(level[2])


@pytest.mark.parametrize("engine", ["c", "python"])
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        pytest.param({"n": "int32"}, {"n": "int32", "j": object}, id="mapping"),
        pytest.param(
            defaultdict(lambda: "str", {"n": "int32"}),
            {"n": "int32", "j": object, "other": "str"},
            id="defaultdict",
        ),
        pytest.param("str", {"n": "str", "j": object}, id="scalar"),
    ],
)
def test_read_csv_json_with_dtype(engine, dtype, expected):
    """A dtype given to execute() still applies to the other columns."""
    df, options = _read_csv_result(
        "n,j\n1,2\n3,\n", {"n": "integer", "j": "json"}, engine=engine, dtype=dtype
    )
    assert {key: options["dtype"][key] for key in expected} == expected
    if isinstance(dtype, dict):
        assert "j" not in dtype
    assert df["n"].dtype == expected["n"]
    assert df["j"].tolist() == [2, None]


@pytest.mark.parametrize(
    ("types", "engine", "kwargs"),
    [
        pytest.param({"n": "integer", "j": "integer"}, "c", {}, id="no-json"),
        pytest.param({"n": "integer", "j": "json"}, "c", {}, id="json"),
        pytest.param({"n": "integer", "j": "json"}, "c", {"usecols": ["n"]}, id="json-excluded"),
        pytest.param(
            {"n": "integer", "j": "json"}, "c", {"converters": {}}, id="custom-converters"
        ),
        pytest.param(
            {"n": "integer", "j": "json"},
            "c",
            {"converters": {"j": int}},
            id="custom-json-converter",
        ),
        pytest.param({"n": "integer", "j": "json"}, "python", {}, id="python"),
        pytest.param({"n": "integer", "j": "integer"}, "pyarrow", {}, id="pyarrow"),
    ],
)
def test_read_csv_keeps_low_memory_default(types, engine, kwargs):
    """No column type, converter, or engine sets a low_memory option."""
    df, options = _read_csv_result("n,j\n" + "1,2\n" * 30, types, engine=engine, **kwargs)
    assert options["engine"] == engine
    assert "low_memory" not in options
    assert len(df) == 30


@pytest.mark.parametrize("converters", [{}, {"j": int}])
def test_read_csv_json_custom_converters(converters):
    """Converters given to execute() replace the json conversion."""
    df, options = _read_csv_result(
        "n,j\n1,2\n", {"n": "integer", "j": "json"}, converters=converters
    )
    assert options["converters"] is converters
    assert "j" not in (options["dtype"] or {})
    assert df["j"].tolist() == [2]


def _string_series(values, infer_string):
    """Express the expected string dtype for the active pandas option."""
    return pd.Series(values, dtype="str" if infer_string else object)


def _types_frame(infer_string, parse_time=True):
    """Build the typed literal expectation for the all-types CSV."""
    missing = float("nan")
    return pd.DataFrame(
        {
            "ti": pd.Series([1, None], dtype="Int64"),
            "si": pd.Series([2, None], dtype="Int64"),
            "i": pd.Series([3, None], dtype="Int64"),
            "bi": pd.Series([4, None], dtype="Int64"),
            "r": pd.Series([1.5, missing], dtype="float64"),
            "d": pd.Series([2.25, missing], dtype="float64"),
            "c": _string_series(["ab ", missing], infer_string),
            "v": _string_series(["plain", missing], infer_string),
            "ml": _string_series(['multi\nline "q", x', missing], infer_string),
            "arr": _string_series(["[1, 2]", missing], infer_string),
            "m": _string_series(["{k=1}", missing], infer_string),
            "rw": _string_series(["{a=1, b=x}", missing], infer_string),
            "dt": pd.Series([pd.Timestamp("2024-02-29"), pd.NaT], dtype="datetime64[us]"),
            "ts": pd.Series(
                [pd.Timestamp("2024-02-29 23:59:58.123"), pd.NaT], dtype="datetime64[ns]"
            ),
            "ts6": pd.Series(
                [pd.Timestamp("2024-02-29 23:59:58.123456"), pd.NaT], dtype="datetime64[ns]"
            ),
            # pandas supplies today's date when it parses a time without a date.
            "tm": (
                pd.Series(pd.to_datetime(["12:34:56.789", None]))
                if parse_time
                else _string_series(["12:34:56.789", None], infer_string)
            ),
            "iv": _string_series(["2 00:00:00.000", None], infer_string),
            "nul": pd.Series([missing, missing], dtype="float64"),
            "u": _string_series(["589f6631-9c50-4f58-a121-e2608a04fc64", None], infer_string),
            "empty": _string_series([missing, missing], infer_string),
            "na": _string_series(["NA", missing], infer_string),
        }
    )


@pytest.mark.filterwarnings("ignore:Could not infer format")
@pytest.mark.parametrize("infer_string", [True, False])
@pytest.mark.parametrize(
    ("data", "types", "read_options", "expected_frame", "pandas_columns"),
    [
        pytest.param(
            _TYPES_CSV,
            _TYPES,
            {},
            _types_frame,
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18],
            id="types",
        ),
        pytest.param(
            _TYPES_CSV,
            _TYPES,
            {"dtype_overrides": {"ti": "float32", "v": "category", "missing": "int64"}},
            lambda infer: _types_frame(infer).astype({"ti": "float32", "v": "category"}),
            [0, 1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 18],
            id="dtype",
        ),
        pytest.param(
            _TYPES_CSV,
            _TYPES,
            {"parse_dates": [12, "ts"]},
            lambda infer: _types_frame(infer, parse_time=False),
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18],
            id="parse_dates",
        ),
        pytest.param(
            _TYPES_CSV,
            _TYPES,
            {"dtype_overrides": {"dt": "string"}},
            lambda infer: _types_frame(infer).assign(
                dt=pd.Series(["2024-02-29", None], dtype="string")
            ),
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18],
            id="dtype_of_date_column",
        ),
        pytest.param(
            '"x","d"\n"1","2024-01-01"\n,\n',
            {"x": "integer", "d": "date"},
            {"dtype": {"x": None}},
            lambda infer: pd.DataFrame(
                {
                    "x": [1.0, float("nan")],
                    "d": pd.Series([pd.Timestamp("2024-01-01"), pd.NaT], dtype="datetime64[us]"),
                }
            ),
            [0, 1],
            id="dtype_none",
        ),
        pytest.param(
            '"x","x","d"\n"1","2","2024-01-01"\n,,\n',
            {"x": "integer", "d": "date"},
            {},
            lambda infer: pd.concat(
                [
                    pd.Series([1, None], name="x", dtype="Int64"),
                    pd.Series([2, None], name="x", dtype="Int64"),
                    pd.Series(
                        [pd.Timestamp("2024-01-01"), pd.NaT], name="d", dtype="datetime64[us]"
                    ),
                ],
                axis=1,
            ),
            [0, 1, 2],
            id="duplicate_names",
        ),
        pytest.param(
            '"v","n"\n"1","1"\n,\n"nan","3"\n"007","4"\n"1e3","5"\n',
            {"v": "varchar", "n": "integer"},
            {},
            lambda infer: pd.DataFrame(
                {
                    "v": _string_series(["1", float("nan"), "nan", "007", "1e3"], infer),
                    "n": pd.Series([1, None, 3, 4, 5], dtype="Int64"),
                }
            ),
            [1],
            id="numeric_looking_strings",
        ),
        pytest.param(
            '"v","w","x"\n"007","a","1"\n,,"2"\n',
            {"x": "integer"},
            {
                "dtype": {
                    "v": pd.ArrowDtype(pa.string()),
                    "w": pd.ArrowDtype(pa.large_string()),
                    "x": pd.Int64Dtype(),
                }
            },
            lambda infer: pd.DataFrame(
                {
                    "v": pd.Series(["007", None], dtype=pd.ArrowDtype(pa.string())),
                    "w": pd.Series(["a", None], dtype=pd.ArrowDtype(pa.large_string())),
                    "x": pd.Series([1, 2], dtype="Int64"),
                }
            ),
            [2],
            id="arrow_string_dtypes",
        ),
        pytest.param(
            "001\t2\t003\n004\t5\t\n",
            {"v": "varchar"},
            {"tab_separated": True},
            lambda infer: pd.DataFrame(
                {"0": [1, 4], "1": [2, 5], "v": _string_series(["3", float("nan")], infer)}
            ),
            [0, 1],
            id="tab_separated_numeric_fields",
        ),
        pytest.param(
            '"v","n"\n"007","1"\n',
            {"v": "varchar", "n": "integer"},
            {"dtype": {"v": str, 0: str}},
            lambda infer: pd.DataFrame({"v": _string_series(["007"], infer), "n": [1]}),
            [1],
            id="dtype_position_key",
        ),
        pytest.param(
            '"v"\n"plain"\n"2024-01-01"\n\n',
            {"v": "varchar"},
            {"parse_dates": ["v"]},
            lambda infer: pd.DataFrame(
                {"v": _string_series(["plain", "2024-01-01", float("nan")], infer)}
            ),
            [],
            id="unparsed_dates",
        ),
        pytest.param(
            "id    \tint    \t    \nname  \tstring \t    \n",
            {"col_name": "varchar"},
            {"tab_separated": True},
            lambda infer: pd.DataFrame(
                {
                    "0": _string_series(["id    ", "name  "], infer),
                    "1": _string_series(["int    ", "string "], infer),
                    "col_name": _string_series(["    ", "    "], infer),
                }
            ),
            [0, 1, 2],
            id="tab_separated_extra_fields",
        ),
        pytest.param(
            "x\t1\t2024-01-01\n\t\t\ny y\t3\t2024-01-02\n",
            {"a": "varchar", "b": "bigint", "c": "date"},
            {"tab_separated": True},
            lambda infer: pd.DataFrame(
                {
                    "a": _string_series(["x", float("nan"), "y y"], infer),
                    "b": pd.Series([1, None, 3], dtype="Int64"),
                    "c": pd.Series(
                        [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-02")],
                        dtype="datetime64[us]",
                    ),
                }
            ),
            [1, 2],
            id="tab_separated",
        ),
    ],
)
def test_read_csv_with_pyarrow_matches_pandas(
    data, types, read_options, expected_frame, pandas_columns, infer_string
):
    """CSV results match literal expectations and pandas where their contracts agree."""
    with pd.option_context("future.infer_string", infer_string):
        read_csv_kwargs = _pyarrow_read_csv_kwargs(types, **read_options)
        expected = expected_frame(infer_string)
        actual = _read_csv_with_pyarrow(io.BytesIO(data.encode()), read_csv_kwargs)
        assert_frame_equal(actual, expected, check_exact=True)
        # Explicit positions retain duplicate names and headerless-column parity.
        # Mapped string columns use PyAthena's preservation contract instead.
        if pandas_columns:
            reference = pd.read_csv(
                io.BytesIO(data.encode()),
                **{**read_csv_kwargs, "dtype": dict(read_csv_kwargs["dtype"])},
            )
            assert_frame_equal(
                actual.iloc[:, pandas_columns], reference.iloc[:, pandas_columns], check_exact=True
            )


@pytest.mark.parametrize("infer_string", [True, False])
def test_read_csv_with_pyarrow_string_dtype_after_dates(infer_string):
    # A string dtype applies again after parse_dates, as with pandas' PyArrow
    # engine, and keeps NULL missing.
    with pd.option_context("future.infer_string", infer_string):
        df = _read_csv_with_pyarrow(
            io.BytesIO(b'"v"\n"2024-01-01"\n\n'),
            _pyarrow_read_csv_kwargs({"v": "varchar"}, parse_dates=["v"]),
        )
    assert df["v"].tolist()[0] == "2024-01-01"
    assert pd.isna(df["v"].tolist()[1])


def test_read_csv_with_pyarrow_ignores_unused_dtype_entries():
    # As with pandas' PyArrow engine, a dtype entry for a column that is not in
    # the result is not validated.
    read_csv_kwargs = _pyarrow_read_csv_kwargs(
        {"v": "varchar"},
        dtype={"v": str, "unused": "not-a-dtype", "unsupported": "decimal128(10, 2)[pyarrow]"},
    )
    data = b'"v"\n"007"\n'
    actual = _read_csv_with_pyarrow(
        io.BytesIO(data), {**read_csv_kwargs, "dtype": dict(read_csv_kwargs["dtype"])}
    )
    assert actual.columns.tolist() == ["v"]
    assert actual["v"].tolist() == ["007"]


def test_read_csv_with_pyarrow_multiline_values_across_blocks():
    # The 2.4 MB file spans several 1 MiB pyarrow read blocks, and its values
    # contain newlines and quotes.
    rows = [(i, f'{"x" * 600}\n"{"y" * 598}"' if i % 2 else "z") for i in range(4000)]
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["id", "v"])
    writer.writerows(rows)
    df = _read_csv_with_pyarrow(
        io.BytesIO(buffer.getvalue().encode()),
        _pyarrow_read_csv_kwargs({"id": "integer", "v": "varchar"}),
    )
    assert df["id"].tolist() == [i for i, _ in rows]
    assert df["v"].tolist() == [v for _, v in rows]
