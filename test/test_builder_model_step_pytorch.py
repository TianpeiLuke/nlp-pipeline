import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile

from sagemaker.workflow.pipeline_context import PipelineSession, _StepArguments
from sagemaker.workflow.properties import Properties
from sagemaker.pytorch import PyTorchModel
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.model_step import ModelStep


from pipelines.config_model_step_pytorch import PytorchModelCreationConfig
from pipelines.builder_model_step_pytorch import PytorchModelStepBuilder


class TestPytorchModelStepBuilder(unittest.TestCase):
    """Unit-tests for PytorchModelStepBuilder"""

    def setUp(self):
        # -------- temporary source dir with dummy inference script ----------
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.src_dir = Path(self.tmp_dir.name)
        (self.src_dir / "inference_script.py").touch()

        # -------- minimal concrete config ----------------------------------
        self.cfg = PytorchModelCreationConfig(
            pipeline_name="MyPipeline",
            region="NA",
            current_date="2025-05-20",
            source_dir=str(self.src_dir),
            inference_entry_point="inference_script.py",
        )

        # -------- common mocks ---------------------------------------------
        self.sm_session = Mock(spec=PipelineSession)
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"

        self.builder = PytorchModelStepBuilder(
            config=self.cfg,
            sagemaker_session=self.sm_session,
            role=self.role,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    # --------------------------------------------------------------------- #
    # helper: construct valid StepArguments expected by ModelStep validator #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _dummy_step_args() -> _StepArguments:
        return _StepArguments(
            request_dict={},      # validator ignores contents for unit test
            job=None,
            method_name="create",
            caller_name="sagemaker.pytorch.model.PyTorchModel",
        )

    # --------------------------- individual tests ------------------------ #
    def test_create_env_config(self):
        env = self.builder._create_env_config()
        self.assertEqual(env["SAGEMAKER_PROGRAM"], "inference_script.py")
        self.assertEqual(env["AWS_REGION"], self.builder.aws_region)

    def test_reject_empty_entry_point(self):
        bad_cfg = self.cfg.model_copy(update={"inference_entry_point": ""})
        with self.assertRaises(ValueError):
            PytorchModelStepBuilder(
                config=bad_cfg, sagemaker_session=self.sm_session, role=self.role
            )

    @patch("pipelines.builder_model_step_pytorch.image_uris.retrieve")
    def test_get_image_uri(self, mock_retrieve):
        mock_retrieve.return_value = "dummy-uri"
        uri = self.builder._get_image_uri()
        self.assertEqual(uri, "dummy-uri")


if __name__ == "__main__":
    unittest.main()