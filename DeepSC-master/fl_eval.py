# fl_eval.py
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import SNR_to_noise, greedy_decode, SeqtoText, BleuScore
from fl_data import EurDatasetLocal, collate_data
from models.transceiver import DeepSC


@torch.no_grad()
def evaluate_bleu(model, data_loader, idx_to_token, pad_idx, start_idx, end_idx, channel: str, snr_db: float, max_len: int = 30):
    model.eval()
    n_var = SNR_to_noise(snr_db)
    seq_to_text = SeqtoText(idx_to_token, end_idx)

    # BLEU-1 (you can also do BLEU-4 by setting weights)
    bleu = BleuScore(1.0, 0.0, 0.0, 0.0)

    scores = []
    pbar = tqdm(data_loader, desc=f"Eval BLEU @ SNR={snr_db}dB", leave=False)

    for sents in pbar:
        sents = sents.to(next(model.parameters()).device)
        pred = greedy_decode(
            model=model,
            src=sents,
            n_var=n_var,
            max_len=max_len,
            padding_idx=pad_idx,
            start_symbol=start_idx,
            channel=channel
        )

        # Convert tokens -> text for BLEU computation
        for gt_tokens, pred_tokens in zip(sents, pred):
            gt_sent = seq_to_text.sequence_to_text(gt_tokens.tolist())
            pd_sent = seq_to_text.sequence_to_text(pred_tokens.tolist())
            if len(gt_sent.strip()) == 0 or len(pd_sent.strip()) == 0:
                continue
            scores.append(bleu.compute_blue_score([gt_sent], [pd_sent])[0])

    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def main():
    """
    Simple CLI to evaluate a trained federated DeepSC model on the test split
    using BLEU-1, matching the defaults from fl_train.py.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing europarl/train_data.pkl etc.")
    parser.add_argument("--vocab_file", type=str, default="europarl/vocab.json",
                        help="Path relative to data_root or absolute")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to a trained DeepSC checkpoint (.pth)")

    # Model / architecture (must match training)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=30)

    # Channel / evaluation SNR
    parser.add_argument("--channel", type=str, default="Rayleigh",
                        choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr_eval", type=float, default=6.0,
                        help="Evaluation SNR in dB")

    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    # Resolve vocab path
    vocab_path = args.vocab_file
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join(args.data_root, vocab_path)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Could not find vocab file: {vocab_path}")

    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    token_to_idx = vocab["token_to_idx"]
    # Build reverse mapping idx -> token
    idx_to_token = {int(idx): tok for tok, idx in token_to_idx.items()}

    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    num_vocab = len(token_to_idx)

    # Load dataset / dataloader
    test_set = EurDatasetLocal(args.data_root, split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_data,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Recreate model (must match training configuration)
    model = DeepSC(
        args.num_layers,
        num_vocab, num_vocab, num_vocab, num_vocab,
        args.d_model,
        args.num_heads,
        args.dff,
        args.dropout,
    ).to(device)

    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)

    bleu = evaluate_bleu(
        model=model,
        data_loader=test_loader,
        idx_to_token=idx_to_token,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        channel=args.channel,
        snr_db=args.snr_eval,
        max_len=args.max_len,
    )

    print(f"Test BLEU-1 @ {args.snr_eval} dB ({args.channel}): {bleu:.4f}")


if __name__ == "__main__":
    main()
