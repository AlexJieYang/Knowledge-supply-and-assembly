"""Portable input resolution; runnable with Python's standard library alone."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest

from assembly.paths import input_path, project_path


class ConfigurationPathsTest(unittest.TestCase):
    def test_inputs_follow_configuration_from_another_working_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'project with spaces'
            expected = project / 'data' / 'core' / 'papers.parquet'
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b'project input')
            caller = root / 'caller'
            decoy = caller / 'data' / 'core' / 'papers.parquet'
            decoy.parent.mkdir(parents=True)
            decoy.write_bytes(b'unrelated input')
            settings = {'research_dir': 'data', 'paths': {'papers': 'core/papers.parquet'}}
            config_file = project / 'config.json'
            config_file.write_text(json.dumps(settings), encoding='utf-8')
            settings = json.loads(config_file.read_text(encoding='utf-8'))
            settings['_project_dir'] = str(config_file.parent)
            previous = Path.cwd()
            try:
                os.chdir(caller)
                self.assertEqual(input_path(settings, 'papers').read_bytes(), b'project input')
                self.assertEqual(project_path(settings, 'results'), project / 'results')
            finally:
                os.chdir(previous)

    def test_explicit_input_override_is_preserved(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / 'external.parquet'
            target.write_bytes(b'external input')
            settings = {'research_dir': 'data', 'paths': {'papers': str(target)}}
            self.assertEqual(input_path(settings, 'papers'), target)
            self.assertEqual(input_path(settings, 'papers').read_bytes(), b'external input')

    def test_relative_embedding_files_follow_configuration(self):
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            target = project / 'embeddings' / 'vectors.npy'
            target.parent.mkdir()
            target.write_bytes(b'embedding input')
            settings = {'_project_dir': str(project)}
            self.assertEqual(project_path(settings, 'embeddings/vectors.npy').read_bytes(), b'embedding input')


if __name__ == '__main__':
    unittest.main()
