# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
import os
from pathlib import Path

# Third Party
# CuRobo
from curobo._src.util.config_io import (
    copy_file_to_path,
    create_dir_if_not_exists,
    file_exists,
    get_filename,
    get_files_from_dir,
    get_path_of_dir,
    is_file_xrdf,
    is_platform_linux,
    is_platform_windows,
    join_path,
    load_yaml,
    merge_dict_a_into_b,
    write_yaml,
)


class TestJoinPath:
    def test_join_path_strings(self):
        result = join_path("foo", "bar")
        assert "foo" in result and "bar" in result

    def test_join_path_with_path_objects(self):
        result = join_path(Path("foo"), Path("bar"))
        assert "foo" in result and "bar" in result

    def test_join_path_with_leading_slash_in_second_arg(self):
        """Test that leading slash in second argument makes it absolute (standard behavior)."""
        result = join_path("/world", "/obstacles")
        # Standard os.path.join behavior: absolute path2 ignores path1
        assert result == "/obstacles"

    def test_join_path_with_trailing_slash_in_first_arg(self):
        """Test that trailing slash in first argument is handled correctly."""
        result = join_path("/world/", "obstacles")
        assert result == os.path.join("/world/", "obstacles")

    def test_join_path_both_slashes(self):
        """Test with trailing slash in first and leading in second (absolute path2)."""
        result = join_path("/world/", "/obstacles")
        # Standard os.path.join behavior: absolute path2 ignores path1
        assert result == "/obstacles"

    def test_join_path_no_slashes(self):
        """Test normal case without any problematic slashes."""
        result = join_path("world", "obstacles")
        assert result == os.path.join("world", "obstacles")

    def test_join_path_empty_strings(self):
        """Test with empty strings."""
        result = join_path("", "obstacles")
        assert "obstacles" in result

    def test_join_path_nested_paths(self):
        """Test joining nested paths."""
        result = join_path("/world/robot", "link1")
        assert result == os.path.join("/world/robot", "link1")


class TestYaml:
    def test_load_yaml_from_file(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnum: 42\n")
        result = load_yaml(str(yaml_file))
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_load_yaml_from_dict(self):
        data = {"key": "value", "num": 42}
        result = load_yaml(data)
        assert result == data

    def test_write_yaml(self, tmp_path):
        yaml_file = tmp_path / "output.yaml"
        data = {"key": "value", "num": 42}
        write_yaml(data, str(yaml_file))
        assert yaml_file.exists()
        loaded = load_yaml(str(yaml_file))
        assert loaded == data


class TestFileCopy:
    def test_copy_file_to_path(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest_dir = tmp_path / "dest"
        result = copy_file_to_path(str(source), str(dest_dir))
        assert os.path.exists(result)
        with open(result) as f:
            assert f.read() == "test content"


class TestGetFilename:
    def test_get_filename_with_extension(self):
        result = get_filename("/path/to/file.txt")
        assert result == "file.txt"

    def test_get_filename_without_extension(self):
        result = get_filename("/path/to/file.txt", remove_extension=True)
        assert result == "file"


class TestGetPathOfDir:
    def test_get_path_of_dir(self):
        result = get_path_of_dir("/path/to/file.txt")
        assert result == "/path/to"


class TestGetFilesFromDir:
    def test_get_files_from_dir(self, tmp_path):
        (tmp_path / "test1.txt").write_text("content")
        (tmp_path / "test2.txt").write_text("content")
        (tmp_path / "other.yaml").write_text("content")
        result = get_files_from_dir(str(tmp_path), [".txt"], "test")
        assert len(result) == 2
        assert "test1.txt" in result
        assert "test2.txt" in result


class TestFileExists:
    def test_file_exists_true(self, tmp_path):
        test_file = tmp_path / "exists.txt"
        test_file.write_text("content")
        assert file_exists(str(test_file)) is True

    def test_file_exists_false(self):
        assert file_exists("/nonexistent/path/file.txt") is False

    def test_file_exists_none(self):
        assert file_exists(None) is False


class TestMergeDict:
    def test_merge_dict_simple(self):
        a = {"key1": "new_value"}
        b = {"key1": "old_value", "key2": "value2"}
        result = merge_dict_a_into_b(a, b)
        assert result["key1"] == "new_value"
        assert result["key2"] == "value2"

    def test_merge_dict_nested(self):
        a = {"outer": {"inner": "new_value"}}
        b = {"outer": {"inner": "old_value", "other": "value"}}
        result = merge_dict_a_into_b(a, b)
        assert result["outer"]["inner"] == "new_value"
        assert result["outer"]["other"] == "value"


class TestPlatformChecks:
    def test_is_platform_windows(self):
        result = is_platform_windows()
        assert isinstance(result, bool)

    def test_is_platform_linux(self):
        result = is_platform_linux()
        assert isinstance(result, bool)


class TestIsFileXrdf:
    def test_is_file_xrdf_true(self):
        assert is_file_xrdf("robot.xrdf") is True
        assert is_file_xrdf("ROBOT.XRDF") is True

    def test_is_file_xrdf_false(self):
        assert is_file_xrdf("robot.urdf") is False
        assert is_file_xrdf("robot.xml") is False


class TestCreateDirIfNotExists:
    def test_create_dir(self, tmp_path):
        new_dir = tmp_path / "new_directory"
        create_dir_if_not_exists(str(new_dir))
        assert new_dir.exists()
        assert new_dir.is_dir()

