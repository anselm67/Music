import torch

from kern import NUM_ARTICULATIONS
from scorer import ScorerConfig, ScorerModule
from scorer.scorer_module import group_systems


def _tiny_config() -> ScorerConfig:
    cfg = ScorerConfig()
    cfg.noter.embed_dim = 32
    cfg.noter.num_head = 2
    cfg.noter.num_encoder_layers = 1
    cfg.noter.num_decoder_layers = 1
    cfg.noter.mlp_dim = 64
    cfg.noter.max_seqlen = 8
    cfg.noter.max_chords = 2
    cfg.noter.vocab_size = 12
    cfg.noter.pad_idx = 0
    cfg.freeze_staffer_steps = 0
    return cfg


def test_generate_grouped_handles_a_four_staff_system() -> None:
    # A single 4-staff system decodes in lockstep (all four in one group).
    cfg = _tiny_config()
    module = ScorerModule(cfg).eval()
    K = 4
    crops = torch.rand(K, 1, *cfg.noter.input_shape)
    widths = torch.full((K,), cfg.noter.input_shape[1])
    owners = torch.zeros(K, dtype=torch.long)
    tokens, arts = module._generate_grouped(crops, widths, owners)
    assert tokens.shape[0] == K
    assert arts.shape == (*tokens.shape, NUM_ARTICULATIONS)


def test_group_systems_pads_to_batch_max_not_ceiling() -> None:
    # Two pages: page 0 has a 2-staff system, page 1 a 1-staff system. The batch's
    # largest system is 2, so smax=2 even though max_staves=4.
    assign_q = [torch.zeros(2, dtype=torch.long), torch.zeros(1, dtype=torch.long)]
    sys_ids = [torch.tensor([0, 0]), torch.tensor([0])]
    grouped_idx, stave_mask = group_systems(
        assign_q, sys_ids, max_staves=4, device=torch.device("cpu")
    )
    assert grouped_idx.shape == (2, 2)  # padded to the batch-max (2), not 4
    # Flat indices go page-by-page then GT-stave order: sys 0 = [0, 1], sys 1 = [2].
    assert grouped_idx.tolist() == [[0, 1], [2, 0]]
    assert stave_mask.tolist() == [[True, True], [True, False]]


def test_group_systems_caps_at_max_staves() -> None:
    # A single 5-staff system with max_staves=4 keeps only the first four staves.
    assign_q = [torch.zeros(5, dtype=torch.long)]
    sys_ids = [torch.zeros(5, dtype=torch.long)]
    grouped_idx, stave_mask = group_systems(
        assign_q, sys_ids, max_staves=4, device=torch.device("cpu")
    )
    assert grouped_idx.shape == (1, 4)
    assert stave_mask.all()


def test_group_systems_batch_max_grows_to_widest_system() -> None:
    # A batch containing a 3-staff system pads every system to 3.
    assign_q = [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)]
    sys_ids = [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)]
    grouped_idx, stave_mask = group_systems(
        assign_q, sys_ids, max_staves=4, device=torch.device("cpu")
    )
    assert grouped_idx.shape == (2, 3)
    assert stave_mask.tolist() == [[True, True, True], [True, True, False]]
