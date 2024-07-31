from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Auto_HookPoint.HookedTransformerAdapter import HookedTransformerAdapter
from tests.test_models import MyModule, gpt2_tokenizer
from tests.test_adapter import Config

def test_sae_training_runner_run():
    model = MyModule()
    model = HookedTransformerAdapter(
        model=model,
        tokenizer=gpt2_tokenizer,
        cfg=Config(
            preproc_fn=lambda x: model.model.emb(x), 
            return_type="logits",
            output_logits_soft_cap=0.0,
            normalization_type="expected_average_only_in",
            block_attr="layers",
            embedding_attr="emb",
        )
    )

    cfg = LanguageModelSAERunnerConfig(
        model_name="test_model",
        hook_name="model.emb.hook_point",
        hook_layer=0,
        d_in=10,
        dataset_path="monology/pile-uncopyrighted",
        is_dataset_tokenized=False,
        streaming=True,
        training_tokens=10,
        device="cpu",
        dtype="float32"
    )

    runner = SAETrainingRunner(cfg, override_model=model)
    result = runner.run()
    
    assert result is not None, "SAETrainingRunner.run() should return a result"
