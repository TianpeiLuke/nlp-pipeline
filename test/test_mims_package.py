# test/test_mims_package.py
import unittest
from unittest.mock import patch, MagicMock, call
import os
import tempfile
import shutil
import tarfile
from pathlib import Path

# Import specific functions and constants from the script to be tested
from src.pipeline_scripts.mims_package import (
    ensure_directory,
    copy_file_robust,
    copy_scripts,
    create_tarfile,
    extract_tarfile,
    list_directory_contents,
    main
)

class TestMimsPackagingHelpers(unittest.TestCase):
    """Unit tests for individual helper functions in the packaging script."""

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.base_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_dummy_file(self, path, content="dummy"):
        """Helper to create a dummy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def test_ensure_directory(self):
        """Test the ensure_directory function."""
        new_dir = self.base_dir / "new_dir"
        self.assertFalse(new_dir.exists())
        
        # Test creation
        ensure_directory(new_dir)
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())
        
        # Test on existing directory (should not raise error)
        ensure_directory(new_dir)
        self.assertTrue(new_dir.exists())

    def test_copy_file_robust(self):
        """Test the copy_file_robust function."""
        src_file = self.base_dir / "src" / "file.txt"
        dst_file = self.base_dir / "dst" / "file.txt"
        self._create_dummy_file(src_file, "some content")

        # Test successful copy
        result = copy_file_robust(src_file, dst_file)
        self.assertTrue(result)
        self.assertTrue(dst_file.exists())
        with open(dst_file, "r") as f:
            self.assertEqual(f.read(), "some content")

        # Test copying a non-existent file
        non_existent_src = self.base_dir / "non_existent.txt"
        result_fail = copy_file_robust(non_existent_src, dst_file)
        self.assertFalse(result_fail)

    def test_copy_scripts(self):
        """Test the copy_scripts function."""
        src_dir = self.base_dir / "source_scripts"
        dst_dir = self.base_dir / "dest_scripts"
        
        # Create a nested structure in source
        self._create_dummy_file(src_dir / "main.py")
        self._create_dummy_file(src_dir / "utils" / "helpers.py")
        
        copy_scripts(src_dir, dst_dir)
        
        # Check if the destination has the same structure
        self.assertTrue((dst_dir / "main.py").exists())
        self.assertTrue((dst_dir / "utils" / "helpers.py").exists())

    def test_create_and_extract_tarfile(self):
        """Test create_tarfile and extract_tarfile functions together."""
        # Setup for create_tarfile
        source_dir = self.base_dir / "source_for_tar"
        output_tar_path = self.base_dir / "output.tar.gz"
        
        file1 = source_dir / "model.pth"
        file2 = source_dir / "code" / "inference.py"
        self._create_dummy_file(file1)
        self._create_dummy_file(file2)
        
        items_to_include = [file1, source_dir / "code"]
        create_tarfile(output_tar_path, source_dir, items_to_include)
        
        self.assertTrue(output_tar_path.exists())
        
        # Setup for extract_tarfile
        extract_dir = self.base_dir / "extracted_contents"
        extract_tarfile(output_tar_path, extract_dir)
        
        # Verify extracted contents
        self.assertTrue((extract_dir / "model.pth").exists())
        self.assertTrue((extract_dir / "code" / "inference.py").exists())
        
    @patch('src.pipeline_scripts.mims_package.logger')
    def test_list_directory_contents(self, mock_logger):
        """Test that list_directory_contents logs file and directory names."""
        dir_to_list = self.base_dir / "dir_to_list"
        self._create_dummy_file(dir_to_list / "file1.txt")
        self._create_dummy_file(dir_to_list / "subdir" / "file2.txt")
        
        list_directory_contents(dir_to_list, "Test Directory")
        
        # Check if the logger was called with messages containing the file names
        log_calls = mock_logger.info.call_args_list
        log_text = "".join([str(c) for c in log_calls])
        
        self.assertIn("file1.txt", log_text)
        self.assertIn(os.path.join("subdir", "file2.txt"), log_text)

class TestMimsPackagingMainFlow(unittest.TestCase):
    """Integration-style tests for the main() function of the packaging script."""

    def setUp(self):
        """Set up a temporary directory structure mimicking the SageMaker environment."""
        self.base_dir = Path(tempfile.mkdtemp())
        
        # Create a structure similar to /opt/ml/processing
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.model_input_dir = self.input_dir / "model"
        self.script_input_dir = self.input_dir / "script"
        
        # Working directory for the script to use
        self.working_dir = self.base_dir / "working"

        # Create all directories
        for d in [self.input_dir, self.output_dir, self.model_input_dir, self.script_input_dir, self.working_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_dummy_file(self, path, content="dummy"):
        """Helper to create a dummy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
            
    def _create_dummy_tar(self, tar_path, files_to_add):
        """Helper to create a dummy tar.gz file."""
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path, arcname in files_to_add:
                tar.add(file_path, arcname=arcname)

    def test_full_packaging_flow(self):
        """
        Test the main() function to ensure it correctly packages model and script files.
        """
        # --- 1. Setup: Use patch.object to replace the script's global Path constants ---
        from src.pipeline_scripts import mims_package
        with patch.object(mims_package, 'MODEL_PATH', self.model_input_dir), \
             patch.object(mims_package, 'SCRIPT_PATH', self.script_input_dir), \
             patch.object(mims_package, 'OUTPUT_PATH', self.output_dir), \
             patch.object(mims_package, 'WORKING_DIRECTORY', self.working_dir):

            # --- 2. Arrange: Create dummy input files ---
            dummy_model_file = self.base_dir / "model.pth"
            dummy_onnx_file = self.base_dir / "model.onnx"
            self._create_dummy_file(dummy_model_file, "pytorch model")
            self._create_dummy_file(dummy_onnx_file, "onnx model")
            
            input_tar_path = self.model_input_dir / "model.tar.gz"
            self._create_dummy_tar(input_tar_path, [
                (dummy_model_file, "model.pth"),
                (dummy_onnx_file, "model.onnx")
            ])

            self._create_dummy_file(self.script_input_dir / "inference.py", "import torch")
            (self.script_input_dir / "subfolder").mkdir()
            self._create_dummy_file(self.script_input_dir / "subfolder" / "utils.py", "def helper(): pass")
            
            # --- 3. Act: Run the main packaging function ---
            main()

            # --- 4. Assert: Check if the output is correct ---
            final_output_tar = self.output_dir / "model.tar.gz"
            self.assertTrue(final_output_tar.exists(), "Final model.tar.gz was not created.")

            with tarfile.open(final_output_tar, "r:gz") as tar:
                tar_contents = tar.getnames()
                self.assertIn("model.pth", tar_contents)
                self.assertIn("model.onnx", tar_contents)
                self.assertIn("code/inference.py", tar_contents)
                self.assertIn("code/subfolder/utils.py", tar_contents)

    def test_no_input_tar(self):
        """Test the scenario where model files are provided directly, not in a tarball."""
        from src.pipeline_scripts import mims_package
        with patch.object(mims_package, 'MODEL_PATH', self.model_input_dir), \
             patch.object(mims_package, 'SCRIPT_PATH', self.script_input_dir), \
             patch.object(mims_package, 'OUTPUT_PATH', self.output_dir), \
             patch.object(mims_package, 'WORKING_DIRECTORY', self.working_dir):

            # Arrange: Create model files directly in the model input directory
            self._create_dummy_file(self.model_input_dir / "model.pth", "direct model")
            self._create_dummy_file(self.script_input_dir / "inference.py", "direct script")
            
            # Act: Run the main function
            main()
            
            # Assert: Check the output tar
            final_output_tar = self.output_dir / "model.tar.gz"
            self.assertTrue(final_output_tar.exists())

            with tarfile.open(final_output_tar, "r:gz") as tar:
                tar_contents = tar.getnames()
                self.assertIn("model.pth", tar_contents)
                self.assertIn("code/inference.py", tar_contents)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)