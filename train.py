import numpy as np
np.random.seed(42)
import pandas as pd

from Model import *
import argparse

def nmse_db(x):
    return 10 * np.log10(x)

def main_train(config, continue_= None):
    
    config['name_train'] = 'data/snr' + str(int(config['snr'])) + '/' + config['split_type'] + '_' + str(config['user_noise']) + '/' + os.path.basename(config['name_train'])
    config['name_val'] = 'data/snr' + str(int(config['snr'])) + '/' + config['split_type'] + '_' + str(config['user_noise']) + '/' + os.path.basename(config['name_val'])

    os.makedirs(os.path.dirname(config['name_val']), exist_ok=True)

    # Check if files exist
    for file_key in ['smomp_file', 'accurate_file', 'user_positions_file', 'rss_image_path']:
        if not os.path.exists(config[file_key]):
            print(f"Error: {config[file_key]} not found!")
    
    print(f"Using device: {config['device']}")
    
    # Initialize RSS processor
    rss_processor = RSSMapProcessor(
        image_path=config['rss_image_path'],
        bs_pixel_coords=config['bs_pixel_coords'],
        bs_real_coords=config['bs_real_coords'],
        image_width_meters=config['image_width_meters']
    )
    print("outside rss")
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config['smomp_file'], config['accurate_file'], 
        config['user_positions_file'], config['split_type'], config['user_noise'], rss_processor)
    
    print("outside load")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2, persistent_workers=True,prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=2, persistent_workers=True,prefetch_factor=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2, persistent_workers=True,prefetch_factor=2, pin_memory=True)
    print("outside load")
    # Initialize model
    #model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Train model
    print("\nStarting training...", flush=True)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=config['epochs'], 
        lr=config['learning_rate'],
        device=config['device'], 
        model_name_val = config['name_val'],
        model_name_train = config['name_train'], continue_ = continue_
        )
    
    # Load best model and evaluate on test set
    # print("\nEvaluating on test set on best val...")
    # model.load_state_dict(torch.load(config['name_val']))
    # test_nmse_val = evaluate_test_set(model, test_loader, device=config['device'])
    
    # print(f"\nFinal Test NMSE: {test_nmse_val:.6f}")
    # print(f"Test NMSE in dB: {10 * np.log10(test_nmse_val):.2f} dB")

    # print("\nEvaluating on test set on best train...")
    # checkpoint = torch.load(config['name_train'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # test_nmse_train = evaluate_test_set(model, test_loader, device=config['device'])
    
    # print(f"\nFinal Test NMSE: {test_nmse_train:.6f}")
    # print(f"Test NMSE in dB: {10 * np.log10(test_nmse_train):.2f} dB")
    
    # print("\nEvaluating on train set on best val...")
    # model.load_state_dict(torch.load(config['name_val']))
    # train_nmse = evaluate_test_set(model, train_loader, device=config['device'])
    
    # print(f"\nFinal Train NMSE: {train_nmse:.6f}")
    # print(f"Test NMSE in dB: {10 * np.log10(train_nmse):.2f} dB")
    
    model_val = model
    model_train = model
    model_val.load_state_dict(torch.load(config['name_val']))

    # Best training model
    checkpoint = torch.load(config['name_train'])
    model_train.load_state_dict(checkpoint['model_state_dict'])
    
    results = {
        "Train Set": {
            "Best Val": nmse_db(evaluate_test_set(model_val, train_loader, device=config['device'])),
            "Best Train": nmse_db(evaluate_test_set(model_train, train_loader, device=config['device'])),
        },
        "Test Set": {
            "Best Val": nmse_db(evaluate_test_set(model_val, test_loader, device=config['device'])),
            "Best Train": nmse_db(evaluate_test_set(model_train, test_loader, device=config['device'])),
        }
    }
    
    print("\nNMSE (dB) Results:\n")
    print(f"{'':<12}{'Best Val':>15}{'Best Train':>15}")
    print("-" * 42)

    for row in results:
        print(f"{row:<12}{results[row]['Best Val']:>15.2f}{results[row]['Best Train']:>15.2f}")

    row = {
        "snr": config["snr"],
        "split_type": config["split_type"],
        "user_noise": config["user_noise"],
        # "train_nmse": 10 * np.log10(train_nmse),
        # "test_nmse_val": 10 * np.log10(test_nmse_train),
        # "test_nmse_train": 10 * np.log10(test_nmse_train),
        "train_nmse": results["Train Set"]["Best Val"],
        "test_nmse_val": results["Test Set"]["Best Val"],
        "test_nmse_train": results["Test Set"]["Best Train"]
    }
    csv_path = 'data/results_pinn.csv'
    df_row = pd.DataFrame([row])

    # if file doesn't exist → write header, else append
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
        
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    # log scale for better visibility
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    
    plt.tight_layout()
    plt.savefig('data/snr' + str(int(config['snr'])) + '/' + config['split_type'] + '_' + str(config['user_noise']) + '/' + 'training_curves.png')
    plt.show()

    return model, val_loader, test_loader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--smomp_file',           type=str,   default='Dataset/initial_estimate_ls_snr0.npy')
    parser.add_argument('--accurate_file',         type=str,   default='Dataset/3D_channel_15GHz_2x2_Pt50.npy')
    parser.add_argument('--user_positions_file',   type=str,   default='Dataset/ue_positions_noisy.txt')
    parser.add_argument('--rss_image_path',        type=str,   default='Dataset/50_15GHz.jpg')
    parser.add_argument('--bs_pixel_coords',       type=int,   nargs=2, default=[287, 293])
    parser.add_argument('--bs_real_coords',        type=float, nargs=2, default=[71.06, 246.29])
    parser.add_argument('--image_width_meters',    type=float, default=527.5)
    parser.add_argument('--batch_size',            type=int,   default=32)
    parser.add_argument('--epochs',                type=int,   default=500)
    parser.add_argument('--learning_rate',         type=float, default=1e-3)
    parser.add_argument('--device',                type=str,   default='cuda')
    parser.add_argument('--name_val',              type=str,   default='simple_ls_val.pth')
    parser.add_argument('--name_train',            type=str,   default='simple_ls_train.pth')
    parser.add_argument('--split_type',            type=str,   default='loc')
    parser.add_argument('--user_noise',            type=float, default=1.0)
    parser.add_argument('--snr',                    type=float, default=0.0)
    parser.add_argument('--continue_training',     action='store_true')
    args = parser.parse_args()

    config = {
        'smomp_file':           args.smomp_file,
        'accurate_file':        args.accurate_file,
        'user_positions_file':  args.user_positions_file,
        'rss_image_path':       args.rss_image_path,
        'bs_pixel_coords':      tuple(args.bs_pixel_coords),
        'bs_real_coords':       tuple(args.bs_real_coords),
        'image_width_meters':   args.image_width_meters,
        'batch_size':           args.batch_size,
        'epochs':               args.epochs,
        'learning_rate':        args.learning_rate,
        'device':               args.device,
        'name_val':             args.name_val,
        'name_train':           args.name_train,
        'split_type':           args.split_type,
        'user_noise':           args.user_noise,
        'snr':                  args.snr
    }

    model = main_train(config, continue_=args.continue_training)
    # config = {
    #     'smomp_file': 'Dataset/initial_estimate_ls_snr0.npy',
    #     'accurate_file': 'Dataset/3D_channel_15GHz_2x2_Pt50.npy',
    #     'user_positions_file': 'Dataset/ue_positions_noisy.txt',
    #     'rss_image_path': 'Dataset/50_15GHz.jpg',
    #     'bs_pixel_coords': (287, 293),
    #     'bs_real_coords': (71.06, 246.29),
    #     'image_width_meters': 527.5,
    #     'batch_size': 32,
    #     'epochs': 500,
    #     'learning_rate': 1e-3,
    #     'device': 'cuda',
    #     'name_val':'simple_ls_0_val.pth',
    #     'name_train':'simple_ls_0_train.pth',
    #     'split_type': 'loc',
    #     'user_noise': 1 # mention standard deviation
    # }
    # model = main_train(config, continue_=True)
