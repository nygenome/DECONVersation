import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('scanpy')

from deconversation import embeddings
from deconversation.core import deconverse
from deconversation.cli import build_parser


@pytest.fixture
def fake_scconcept(monkeypatch):
    calls = []
    class Concept:
        def __init__(self, cache_dir):
            calls.append(('init', cache_dir))
        def load_config_and_model(self, model_name):
            calls.append(('load', model_name))
        def extract_embeddings(self, adata, species, batch_size, num_workers):
            calls.append(('extract', adata.var_names.tolist(), species, batch_size, num_workers))
            return {'cls_cell_emb': adata.X[:, :2]}
    module = types.ModuleType('concept')
    module.scConcept = Concept
    monkeypatch.setitem(sys.modules, 'concept', module)
    return calls


def test_scconcept_extracts_both_inputs_with_one_model(fake_scconcept):
    bulk = pd.DataFrame([[3., 1.]], index=['sample'], columns=['ENSG1.1', 'ENSG2'])
    sig = pd.DataFrame([[1., 0.], [0., 1.]], index=['A', 'B'], columns=['ENSG1', 'ENSG2'])
    b, s = embeddings.get_scconcept_embeddings(bulk, sig, 'checkpoint', 'cache', batch_size=2)
    assert list(b.index) == ['sample'] and list(s.index) == ['A', 'B']
    assert list(b.columns) == list(s.columns) == ['scConcept_0', 'scConcept_1']
    assert [x[0] for x in fake_scconcept].count('load') == 1
    extracted = [x for x in fake_scconcept if x[0] == 'extract']
    assert extracted == [('extract', ['ENSG1', 'ENSG2'], 'hsapiens', 2, 0)] * 2


@pytest.mark.parametrize('genes,values,message', [
    (['TP53'], [[1]], 'Ensembl'),
    (['ENSG1', 'ENSG1.2'], [[1, 2]], 'unique'),
    (['ENSG1'], [[-1]], 'nonnegative'),
    (['ENSG1'], [[0]], 'nonzero'),
    (['ENSG1'], [[float('nan')]], 'finite'),
])
def test_scconcept_rejects_invalid_inputs(genes, values, message):
    with pytest.raises(ValueError, match=message):
        embeddings.get_embedding_scconcept(pd.DataFrame(values, columns=genes), 'model', 'cache')


def test_scconcept_deconverse_runs_nnls(tmp_path, fake_scconcept):
    sig = pd.DataFrame({'A': [1., 0.], 'B': [0., 1.]}, index=['ENSG1', 'ENSG2'])
    bulk = pd.DataFrame({'sample': [3., 1.]}, index=sig.index)
    sig.to_csv(tmp_path / 'sig.csv')
    bulk.to_csv(tmp_path / 'bulk.csv')
    result = deconverse(str(tmp_path / 'bulk.csv'), model='checkpoint', mode='scconcept',
                        sig_df=str(tmp_path / 'sig.csv'), temp_output_dir=str(tmp_path / 'out'))
    np.testing.assert_allclose(result.loc['sample', ['A', 'B']], [.75, .25])
    assert (tmp_path / 'out' / 'bulk_embedding.csv').is_file()
    assert (tmp_path / 'out' / 'signature_embedding.csv').is_file()


def test_scconcept_cli_accepts_model_and_species_options():
    args = build_parser().parse_args(['--mode', 'scconcept', '--model', 'checkpoint',
                                      '--scconcept-species', 'mmusculus',
                                      '--scconcept-cache-dir', '/tmp/models'])
    assert (args.mode, args.model, args.scconcept_species, args.scconcept_cache_dir) == (
        'scconcept', 'checkpoint', 'mmusculus', '/tmp/models')
