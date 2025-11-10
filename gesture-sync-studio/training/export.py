"""
Export trained model for use in Blender addon.
"""

import torch
import argparse
from pathlib import Path
import logging

from model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_dim: int,
    sequence_length: int = 100
):
    """
    Export model to ONNX format.

    Args:
        model: Trained PyTorch model
        output_path: Output ONNX file path
        input_dim: Input feature dimension
        sequence_length: Example sequence length
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, input_dim)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['audio_features'],
        output_names=['gestures'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'sequence_length'},
            'gestures': {0: 'batch_size', 1: 'sequence_length'}
        }
    )

    logger.info(f"Exported model to ONNX: {output_path}")


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_dim: int,
    sequence_length: int = 100
):
    """
    Export model to TorchScript format.

    Args:
        model: Trained PyTorch model
        output_path: Output .pt file path
        input_dim: Input feature dimension
        sequence_length: Example sequence length
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, input_dim)

    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save
    traced_model.save(output_path)

    logger.info(f"Exported model to TorchScript: {output_path}")


def load_and_export(
    checkpoint_path: str,
    output_path: str,
    format: str = 'onnx',
    model_config: dict = None
):
    """
    Load checkpoint and export to specified format.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Output file path
        format: Export format ('onnx' or 'torchscript')
        model_config: Model configuration dict
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    if model_config is None:
        # Default config
        model_config = {
            'type': 'lstm',
            'input_dim': 29,
            'hidden_dim': 256,
            'output_dim': 84,
            'num_layers': 3,
            'dropout': 0.1
        }

    model = create_model(**model_config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Export
    if format == 'onnx':
        export_to_onnx(
            model,
            output_path,
            input_dim=model_config['input_dim']
        )
    elif format == 'torchscript':
        export_to_torchscript(
            model,
            output_path,
            input_dim=model_config['input_dim']
        )
    else:
        raise ValueError(f"Unknown export format: {format}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export trained model for Blender")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['onnx', 'torchscript'],
        default='onnx',
        help='Export format (default: onnx)'
    )

    parser.add_argument(
        '--input-dim',
        type=int,
        default=29,
        help='Input feature dimension'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension'
    )

    parser.add_argument(
        '--output-dim',
        type=int,
        default=84,
        help='Output dimension'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of layers'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='lstm',
        choices=['lstm', 'simple', 'transformer'],
        help='Model type'
    )

    args = parser.parse_args()

    # Build model config
    model_config = {
        'type': args.model_type,
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'num_layers': args.num_layers,
        'dropout': 0.1
    }

    # Export
    load_and_export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        format=args.format,
        model_config=model_config
    )

    logger.info("Export complete!")


if __name__ == '__main__':
    main()
