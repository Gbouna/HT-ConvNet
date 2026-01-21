#! /usr/bin/env python
#! coding:utf-8
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from dataloader.jhmdb_loader import load_jhmdb_data, Jdata_generator, JConfig
from dataloader.shrec_loader import load_shrec_data, Sdata_generator, SConfig
from models.DDNet_Original import DDNet_Original as DDNet
import sys
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    total_inference_time = 0
    start_time = time.time()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data1, data2, target in tqdm(test_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            batch_start_time = time.time()
            output = model(data1, data2)
            batch_end_time = time.time()
            total_inference_time += (batch_end_time - batch_start_time)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    end_time = time.time()

    # Accuracy 
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    # Precision / Recall / F1 (macro) 
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    num_classes = torch.unique(all_targets).numel()
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = ((all_preds == c) & (all_targets == c)).sum().item()
        fp = ((all_preds == c) & (all_targets != c)).sum().item()
        fn = ((all_preds != c) & (all_targets == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    # Output
    print(
        f'Test set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * test_accuracy:.2f}%)'
    )
    print(
        f'Precision (macro): {macro_precision:.4f}, '
        f'Recall (macro): {macro_recall:.4f}, '
        f'F1-score (macro): {macro_f1:.4f}'
    )

    # ---- FPS ----
    total_frames = len(test_loader.dataset)
    total_time = end_time - start_time
    fps = total_frames / total_time
    print(f'Processed {total_frames} frames in {total_time:.2f} seconds ({fps:.2f} FPS)')
    print(f'Total inference time for all batches: {total_inference_time:.2f} seconds')
    print(f'Average inference time per batch: {total_inference_time / len(test_loader):.6f} seconds')


def main():
    # Parse command line arguments for the model path and dataset selection
    parser = argparse.ArgumentParser(description='Test a trained model on a dataset.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--dataset', type=int, required=True, help='0 for JHMDB, 1 for SHREC coarse, 2 for SHREC fine.')
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA testing.')
    args = parser.parse_args()

    # Load the data and initialize the data loaders
    if args.dataset == 0:
        Config = JConfig()
        data_generator = Jdata_generator
        load_data = load_jhmdb_data
        clc_num = Config.clc_num
    elif args.dataset == 1:
        Config = SConfig()
        load_data = load_shrec_data
        clc_num = Config.class_coarse_num
        data_generator = Sdata_generator('coarse_label')
    elif args.dataset == 2:
        Config = SConfig()
        clc_num = Config.class_fine_num
        load_data = load_shrec_data
        data_generator = Sdata_generator('fine_label')
    else:
        print("Unsupported dataset!")
        sys.exit(1)

    # Set device
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    # Load the model
    model = DDNet(Config.frame_l, Config.joint_n, Config.joint_d,
                  Config.feat_d, Config.filters, clc_num).to(device)
    # Load the saved model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Calculate the total number of parameters
    total_params = count_parameters(model)
    print(f'Total number of parameters in the model: {total_params}')

    _, Test, le = load_data()
    data_gen = data_generator(Test, Config, le)

    X_0_t, X_1_t, Y_t = data_gen
    X_0_t = torch.from_numpy(X_0_t).type(torch.FloatTensor)
    X_1_t = torch.from_numpy(X_1_t).type(torch.FloatTensor)
    Y_t = torch.from_numpy(Y_t).type(torch.LongTensor)

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)  # Use an appropriate batch size

    # Run the test
    test(model, device, test_loader)

if __name__ == '__main__':
    main()
