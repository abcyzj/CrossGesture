import configargparse


def get_parser():
    parser = configargparse.ArgParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, help='Baijia')
    parser.add_argument('--joint_num', type=int, required=True)
    parser.add_argument('--joint_repr', type=str, required=True)
    parser.add_argument('--pose_hidden_size', type=int, required=True)
    parser.add_argument('--num_embedding', type=int)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--vae_type', type=str, required=True)
    parser.add_argument('--num_vq_head', type=int, default=1)
    parser.add_argument('--norm_type', type=str, required=True)
    parser.add_argument('--encoder_channels', nargs='+', type=int, default=[32, 32, 64, 128, 128])
    parser.add_argument('--encoder_downsample_layers', nargs='+', type=int, default=[1, 0, 1, 0, 1])
    parser.add_argument('--enc_dilations', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--decoder_channels', nargs='+', type=int, default=[128, 64, 64, 32, 32])
    parser.add_argument('--decoder_upsample_layers', nargs='+', type=int, default=[1, 0, 1, 0, 1])
    parser.add_argument('--dec_dilations', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--seq_stride', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--optim', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resume_vae', type=str)
    parser.add_argument('--resume_prior', type=str)
    parser.add_argument('--lambda_pose', type=float, default=1)
    parser.add_argument('--lambda_kl', type=float, default=1e-4)
    parser.add_argument('--lambda_commit', type=float)
    parser.add_argument('--lambda_ema_decay', type=float)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--audio_latent_dim', type=int, default=128)
    parser.add_argument('--prior_hidden_dim', type=int, default=128)
    parser.add_argument('--num_prior_dec_layer', type=int)

    return parser
