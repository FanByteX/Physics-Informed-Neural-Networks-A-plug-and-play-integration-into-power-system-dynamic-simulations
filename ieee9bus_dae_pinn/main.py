"""
Main entry point for IEEE 9-bus DAE-PINN training
"""
import argparse
import torch
import numpy as np

from trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IEEE 9-bus DAE-PINN Training')
    
    # Data arguments
    parser.add_argument('--num_train', type=int, default=10000,
                        help='Number of training samples (default: 10000)')
    parser.add_argument('--num_test', type=int, default=1000,
                        help='Number of test samples (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: None = full batch)')
    
    # Model arguments
    parser.add_argument('--num_IRK_stages', type=int, default=10,
                        help='Number of IRK stages (default: 10)')
    parser.add_argument('--stacked', action='store_true', default=True,
                        help='Use stacked architecture (separate network per state)')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function: tanh, relu, sin, swish (default: tanh)')
    parser.add_argument('--dyn_layers', type=int, nargs='+', default=[12, 64, 64, 64],
                        help='Dynamic network layer sizes (default: 12 64 64 64)')
    parser.add_argument('--alg_layers', type=int, nargs='+', default=[12, 64, 64],
                        help='Algebraic network layer sizes (default: 12 64 64)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--h', type=float, default=0.04,
                        help='Time step size (default: 0.04)')
    parser.add_argument('--loss_weight_dyn', type=float, default=1.0,
                        help='Weight for dynamic loss (default: 1.0)')
    parser.add_argument('--loss_weight_alg', type=float, default=1.0,
                        help='Weight for algebraic loss (default: 1.0)')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=[None, 'plateau', 'step'],
                        help='Learning rate scheduler (default: None)')
    parser.add_argument('--patience', type=int, default=1000,
                        help='Scheduler patience (default: 1000)')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Scheduler factor (default: 0.5)')
    
    # Output arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory (default: ./logs)')
    parser.add_argument('--model_name', type=str, default='ieee9bus_pinn',
                        help='Model name for saving (default: ieee9bus_pinn)')
    parser.add_argument('--test_every', type=int, default=100,
                        help='Test every N epochs (default: 100)')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save every N epochs (default: 1000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    
    # Config paths
    parser.add_argument('--config_dynamic', type=str,
                        default='../plug/config_files/config_machines_dynamic.yaml',
                        help='Path to dynamic config file')
    parser.add_argument('--config_static', type=str,
                        default='../plug/config_files/config_machines_static.yaml',
                        help='Path to static config file')
    parser.add_argument('--Y_admittance', type=str,
                        default='../plug/config_files/network_admittance.pt',
                        help='Path to admittance matrix')
    parser.add_argument('--use_tqdm', action='store_true', default=True,
                        help='Show tqdm progress bar')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Print configuration
    print("=" * 60)
    print("IEEE 9-Bus DAE-PINN Training Configuration")
    print("=" * 60)
    print(f"Training samples: {args.num_train}")
    print(f"Test samples: {args.num_test}")
    print(f"Batch size: {args.batch_size if args.batch_size else 'full batch'}")
    print(f"IRK stages: {args.num_IRK_stages}")
    print(f"Architecture: {'stacked' if args.stacked else 'combined'}")
    print(f"Activation: {args.activation}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Time step (h): {args.h}")
    print(f"Loss weights: dyn={args.loss_weight_dyn}, alg={args.loss_weight_alg}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Trainer(
        config_dynamic_path=args.config_dynamic,
        config_static_path=args.config_static,
        Y_admittance_path=args.Y_admittance,
        log_dir=args.log_dir,
        device=args.device,
    )
    
    # Setup model
    dyn_layer_size = args.dyn_layers + [args.num_IRK_stages + 1]
    alg_layer_size = args.alg_layers + [18 * (args.num_IRK_stages + 1)]
    
    trainer.setup_model(
        num_IRK_stages=args.num_IRK_stages,
        dyn_layer_size=dyn_layer_size,
        alg_layer_size=alg_layer_size,
        activation=args.activation,
        stacked=args.stacked,
    )
    
    # Setup data
    trainer.setup_data(
        num_train=args.num_train,
        num_test=args.num_test,
        num_IRK_stages=args.num_IRK_stages,
    )
    
    # Setup optimizer
    trainer.setup_optimizer(
        lr=args.lr,
        scheduler_type=args.scheduler,
        patience=args.patience,
        factor=args.factor,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.resume(args.resume)

    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        h=args.h,
        loss_weights=[args.loss_weight_dyn, args.loss_weight_alg],
        test_every=args.test_every,
        save_every=args.save_every,
        model_name=args.model_name,
        use_tqdm=args.use_tqdm,
    )
    
    print("\nTraining finished!")


if __name__ == '__main__':
    main()
