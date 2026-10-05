import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEECH = ROOT / "examples" / "speech-asr"


class ModelQualityBaselineSourceTests(unittest.TestCase):
    def test_initializer_freezes_reviewed_manifests_and_parent(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_quality_baseline.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'TRAIN_MANIFEST_SHA256 = "89a8624a5dc46ef28845f35591fc1e623dfbd3026d7a4729b7578153d03baf5a"',
            source,
        )
        self.assertIn(
            'VALIDATION_MANIFEST_SHA256 = "07ebc41041238c1ec374ad64eefe7209fd6c11d1050e8f6f72f0226d358c8923"',
            source,
        )
        self.assertIn('DEFAULT_PARENT = "exp-a6f83c0451532122"', source)
        self.assertIn('"records") != 125', source)
        self.assertIn('"records") != 95', source)
        self.assertIn('"max_wer": None', source)
        self.assertIn('"max_cer": None', source)
        self.assertIn('"max_realtime_factor": 0.01', source)
        self.assertIn('"max_latency_p95_ms": 25.0', source)

    def test_baseline_keeps_v3_training_policy(self):
        source = (
            SPEECH / "agent" / "init_cnn_ctc_v3_quality_baseline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('int(package["training"]["epochs"])', source)
        self.assertIn('int(package["training"]["batch_size"])', source)
        self.assertIn('float(package["training"]["learning_rate"])', source)
        self.assertIn('"model_id": "cnn_ctc_v3"', source)

    def test_edge_provisioning_is_out_of_band_and_hash_verified(self):
        source = (ROOT / "scripts" / "provision-speech-model-data-edge.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("BatchMode=yes", source)
        self.assertNotIn("StrictHostKeyChecking=no", source)
        self.assertIn("edge-speech-bootstrap.sh", source)
        self.assertIn("rsync -a --delete --checksum", source)
        self.assertIn("prepare_ami.py", source)
        self.assertIn("qualify-speech-model-data.sh", source)
        self.assertIn(
            "edge validation manifest hash differs from reviewed identity",
            source,
        )

    def test_edge_worker_propagates_and_validates_benchmark_identity(self):
        source = (SPEECH / "agent" / "edge_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('manifest["benchmark"]["id"]', source)
        self.assertIn("benchmark id mismatch", source)
        self.assertIn("benchmark manifest hash mismatch", source)

    def test_v3_training_records_processed_audio_budget(self):
        source = (
            SPEECH / "training" / "train_cnn_ctc_v3.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"train_audio_seconds_per_epoch"', source)
        self.assertIn('"processed_train_audio_seconds"', source)
        self.assertIn('"validation_audio_seconds"', source)


if __name__ == "__main__":
    unittest.main()
