# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

from datetime import datetime
from unittest.mock import PropertyMock, patch

import polars as pl
import pytest
from fsspec.implementations.local import LocalFileSystem

from pyathena.error import OperationalError
from pyathena.polars.converter import DefaultPolarsTypeConverter
from pyathena.polars.result_set import AthenaPolarsResultSet, PolarsDataFrameIterator

_ROWS_BEFORE_FAILURE = 300_000


def _chunked_result_set() -> AthenaPolarsResultSet:
    """Create a chunked result set without calling Athena.

    Returns:
        An AthenaPolarsResultSet with only the attributes used by the chunk readers.
    """
    result_set = AthenaPolarsResultSet.__new__(AthenaPolarsResultSet)  # bypass __init__
    result_set._chunksize = 10_000
    result_set._cache_type = None
    result_set._kwargs = {}
    result_set._metadata = None
    return result_set


class TestAthenaPolarsResultSet:
    def test_iter_csv_chunks_raises_when_read_fails_partway(self, tmp_path):
        """A CSV read that fails partway through the data raises instead of ending early."""
        path = tmp_path / "result.csv"
        path.write_text(
            "a\n" + "".join(f"{i}\n" for i in range(_ROWS_BEFORE_FAILURE)) + "not-a-number\n"
        )
        result_set = _chunked_result_set()
        with (
            patch.object(
                AthenaPolarsResultSet,
                "output_location",
                new_callable=PropertyMock,
                return_value=str(path),
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_dtypes",
                new_callable=PropertyMock,
                return_value={"a": pl.Int64},
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_parquet_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(AthenaPolarsResultSet, "_is_csv_readable", return_value=True),
            pytest.raises(OperationalError, match="not-a-number"),
        ):
            list(result_set._iter_csv_chunks())

    @pytest.mark.parametrize(
        "read_options",
        [{}, {"low_memory": True}],
        ids=["parquet-chunk-reader", "scan-parquet"],
    )
    def test_iter_parquet_chunks_raises_when_read_fails_partway(self, tmp_path, read_options):
        """A Parquet read that fails partway through the data raises instead of ending early.

        Without read options, the files of the data manifest are read with
        ParquetChunkReader; with them, Polars scans the unload location.
        """
        pl.DataFrame({"a": range(_ROWS_BEFORE_FAILURE)}).write_parquet(
            tmp_path / "0.parquet", row_group_size=10_000
        )
        valid = (tmp_path / "0.parquet").read_bytes()
        # Keep the footer magic but corrupt the metadata so the second file fails to read.
        (tmp_path / "1.parquet").write_bytes(valid[: len(valid) // 2] + b"\x00" * 64 + valid[-8:])
        result_set = _chunked_result_set()
        result_set._kwargs = dict(read_options)
        result_set._unload_location = f"{tmp_path}/"
        result_set._data_manifest = [str(tmp_path / "0.parquet"), str(tmp_path / "1.parquet")]
        chunks: list[pl.DataFrame] = []
        with (
            patch.object(
                AthenaPolarsResultSet,
                "_parquet_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch("pyathena.polars.result_set.S3FileSystem", lambda **_: LocalFileSystem()),
            patch("polars.scan_parquet", wraps=pl.scan_parquet) as scan_parquet,
            pytest.raises(OperationalError) as error,
        ):
            # extend() keeps the chunks read before the failure.
            chunks.extend(result_set._iter_parquet_chunks())
        # The error comes from the corrupt file, after the valid one was read.
        assert not isinstance(error.value.__cause__, AttributeError)
        assert scan_parquet.called is bool(read_options)
        if not read_options:
            assert sum(chunk.height for chunk in chunks) == _ROWS_BEFORE_FAILURE

    @pytest.mark.parametrize("reader", ["_read_csv", "_iter_csv_chunks"])
    def test_csv_read_kwargs_replace_defaults(self, tmp_path, reader):
        """Read arguments given to execute() replace the ones the result set chooses."""
        path = tmp_path / "result.csv"
        path.write_text("1;x\n2;y\n")
        # Polars names headerless columns from column_1 before 2.0 and from column_0 since.
        first, second = pl.read_csv(b"1;x\n", separator=";", has_header=False).columns
        result_set = _chunked_result_set()
        result_set._kwargs = {
            "separator": ";",
            "has_header": False,
            "schema_overrides": {first: pl.Utf8},
        }
        with (
            patch.object(
                AthenaPolarsResultSet,
                "output_location",
                new_callable=PropertyMock,
                return_value=str(path),
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_dtypes",
                new_callable=PropertyMock,
                return_value={"1;x": pl.Int64},
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_parquet_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(AthenaPolarsResultSet, "_is_csv_readable", return_value=True),
        ):
            result = getattr(result_set, reader)()
            df = result if isinstance(result, pl.DataFrame) else pl.concat(list(result))
        assert df.to_dict(as_series=False) == {first: ["1", "2"], second: ["x", "y"]}

    @pytest.mark.parametrize("reader", ["_read_csv", "_iter_csv_chunks"])
    def test_txt_new_columns_with_schema_overrides(self, tmp_path, reader):
        """schema_overrides given with new_columns reach Polars with them for a .txt file."""
        path = tmp_path / "result.txt"
        path.write_text("001\tx\n")
        result_set = _chunked_result_set()
        # Polars 2.0 requires new_columns to name every column of the file.
        result_set._kwargs = {"new_columns": ["z", "w"], "schema_overrides": {"z": pl.Utf8}}
        with (
            patch.object(
                AthenaPolarsResultSet,
                "output_location",
                new_callable=PropertyMock,
                return_value=str(path),
            ),
            patch.object(AthenaPolarsResultSet, "_get_column_names", return_value=["a", "b"]),
            patch.object(
                AthenaPolarsResultSet, "_csv_dtypes", new_callable=PropertyMock, return_value={}
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_parquet_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
            patch.object(AthenaPolarsResultSet, "_is_csv_readable", return_value=True),
        ):
            result = getattr(result_set, reader)()
            df = result if isinstance(result, pl.DataFrame) else pl.concat(list(result))
        assert df.to_dict(as_series=False) == {"z": ["001"], "w": ["x"]}

    @pytest.mark.parametrize(
        ("reader", "function"),
        [
            ("_iter_csv_chunks", "scan_csv"),
            ("_read_parquet", "read_parquet"),
            ("_iter_parquet_chunks", "scan_parquet"),
            ("_read_parquet_schema", "scan_parquet"),
        ],
    )
    def test_storage_options_replace_defaults(self, reader, function):
        """storage_options given to execute() replace PyAthena's without computing them."""
        result_set = _chunked_result_set()
        result_set._unload_location = "s3://bucket/unload/"
        result_set._kwargs = {"storage_options": {"anon": True}}
        with (
            patch.object(
                AthenaPolarsResultSet,
                "output_location",
                new_callable=PropertyMock,
                return_value="s3://bucket/result.csv",
            ),
            patch.object(
                AthenaPolarsResultSet, "_csv_dtypes", new_callable=PropertyMock, return_value={}
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                side_effect=AssertionError("replaced storage options were computed"),
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_parquet_storage_options",
                new_callable=PropertyMock,
                side_effect=AssertionError("replaced storage options were computed"),
            ),
            patch.object(AthenaPolarsResultSet, "_is_csv_readable", return_value=True),
            patch.object(AthenaPolarsResultSet, "_prepare_parquet_location", return_value=True),
            patch(f"polars.{function}") as read,
            patch("pyathena.polars.result_set.to_column_info"),
        ):
            result = getattr(result_set, reader)()
            if not isinstance(result, (pl.DataFrame, tuple)):
                list(result)
        assert read.call_args.kwargs["storage_options"] == {"anon": True}

    def test_csv_storage_options_replace_defaults(self):
        """storage_options given to execute() open a CSV result through fsspec, not Polars."""
        result_set = _chunked_result_set()
        result_set._kwargs = {"storage_options": {"anon": True}}
        with (
            patch.object(
                AthenaPolarsResultSet,
                "output_location",
                new_callable=PropertyMock,
                return_value="s3://bucket/result.csv",
            ),
            patch.object(
                AthenaPolarsResultSet, "_csv_dtypes", new_callable=PropertyMock, return_value={}
            ),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                side_effect=AssertionError("replaced storage options were computed"),
            ),
            patch.object(AthenaPolarsResultSet, "_is_csv_readable", return_value=True),
            patch("pyathena.polars.result_set.fsspec.open") as open_,
        ):
            open_.return_value.__enter__.return_value.read.return_value = b"a\n1\n"
            df = result_set._read_csv()
        open_.assert_called_once_with("s3://bucket/result.csv", "rb", anon=True)
        assert df.to_dict(as_series=False) == {"a": [1]}

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, {"t": [datetime(2020, 1, 2, 3, 4, 5, 123456), None]}),
            ({"columns": ["v"]}, {}),
            ({"new_columns": ["t2", "v2"]}, {"t2": [datetime(2020, 1, 2, 3, 4, 5, 123456), None]}),
            ({"with_column_names": lambda names: [n.upper() for n in names]}, OperationalError),
        ],
    )
    def test_read_csv_truncates_timestamps(self, kwargs, expected):
        """Timestamps that fail to parse are read again as text and truncated.

        With ``with_column_names`` given to execute(), Polars renames the columns as it
        reads them, so they are not, and the read fails as before. No AWS calls; the
        GetQueryResults rows are mocked.
        """
        result_set = AthenaPolarsResultSet.__new__(AthenaPolarsResultSet)  # bypass __init__
        result_set._query_execution = None
        result_set._converter = DefaultPolarsTypeConverter()
        result_set._kwargs = kwargs
        result_set._metadata = tuple(
            {"Name": n, "Type": t, "Precision": 3, "Scale": 0, "Nullable": "UNKNOWN"}
            for n, t in (("t", "timestamp"), ("v", "varchar"))
        )
        data = b'"t","v"\n"2020-01-02 03:04:05.123456789012","x"\n,"y"\n'
        with (
            patch.object(AthenaPolarsResultSet, "_fetch_all_rows_as_csv", return_value=data),
            patch.object(
                AthenaPolarsResultSet,
                "_csv_storage_options",
                new_callable=PropertyMock,
                return_value={},
            ),
        ):
            if expected is OperationalError:
                with pytest.raises(OperationalError):
                    result_set._read_csv()
                return
            df = result_set._read_csv()
        for name, values in expected.items():
            assert df.schema[name] == pl.Datetime("us")
            assert df[name].to_list() == values
        if not expected:
            assert df.columns == ["v"]


class TestPolarsDataFrameIterator:
    @pytest.mark.parametrize(
        "reader",
        ["dataframe", "generator"],
        ids=["dataframe", "generator"],
    )
    def test_close_stops_iteration(self, reader):
        """A closed iterator yields nothing for either reader kind."""
        reader = (
            pl.DataFrame({"a": [1, 2]})
            if reader == "dataframe"
            else (pl.DataFrame({"a": [1]}) for _ in range(2))
        )
        df_iter = PolarsDataFrameIterator(reader, {}, ["a"])
        df_iter.close()
        assert list(df_iter) == []
