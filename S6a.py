import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.0'
import rpy2.robjects as robjects
from torch.utils.data import DataLoader, TensorDataset
parser = argparse.ArgumentParser(description='Nanopore data')
parser.add_argument('-v','--vali', default='./backup_4T/newaug2.test_matrix.rds', type=str,
                    help='input file name',metavar="character")
parser.add_argument('-t','--train', default='./backup_4T/newaug2.train_matrix.rds', type=str,
                    help='Number of cores allocated',metavar="character")
parser.add_argument('-o','--name', default='./backup_4T/Input_CNN_7pos_Xen_01.out', type=str,
                    help='input file name',metavar="character")
parser.add_argument('-e','--epoch', default=1000, type=int,
                    help='input file name',metavar="integer")
parser.add_argument('-b','--batch', default=256, type=int,
                    help='input file name',metavar="integer")
parser.add_argument('-s','--seed', default=9999, type=int,
                    help='input file name',metavar="integer")
args = parser.parse_args()
if (args.vali is None or args.vali is None):
    print(args.print_help())
    warnings.warn("Validation/testing and training files must be supplied (-v & -t).")



class ResidualModule(nn.Module):
    def __init__(self, input_channels, out_channels=60):
        super(ResidualModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=5,padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=4,padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=2,padding=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        self.locally_connected1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        self.loc_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.loc_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.dropout1 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(60 * 43, 256)  # Assuming input size 60x7x43 after the previous layers
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 512)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, 2)  # Assuming 2 classes

    def forward(self, x):
        x5 = self.conv1(x)
        x4 = self.conv2(x)
        x3 = self.conv3(x)
        x2 = self.conv4(x)
        x2 = self.locally_connected1(x2)
        y = torch.add(x3,x2)

        y3 = self.loc_conv1(y)
        y2 = self.loc_conv2(y)

        z = torch.add(y2,x4)
        z = self.conv5(z)

        final = z+y3+x5

        final = self.dropout1(final)
        final = self.flatten(final)
        final = F.relu(self.fc1(final))
        final = self.dropout2(final)
        final = F.relu(self.fc2(final))
        final = self.dropout3(final)
        final = self.fc3(final)
        return F.softmax(final, dim=1)
def accuracy(logits, y_true):
    _, indices = torch.max(logits, 1)
    correct_samples = torch.sum(indices == y_true)
    return float(correct_samples) / y_true.shape[0]
def main():
    rds_train_data = robjects.r['readRDS'](args.train)  # Replace 'file.rds' with your file path
    train_df = robjects.conversion.rpy2py(rds_train_data)
    # train_matrix_x= np.transpose(np.array(train_df[1]), (0, 3, 1, 2))[0:500000]
    train_matrix_x = np.array(train_df[1])
    # x=np.isnan(train_matrix_x).any()
    #normalization
    reshaped_data = train_matrix_x.reshape(train_matrix_x.shape[0], -1)
    normalized_data = (reshaped_data - reshaped_data.min(axis=0)) / (
                reshaped_data.max(axis=0) - reshaped_data.min(axis=0) + 1e-8)
    train_matrix_x = normalized_data.reshape(train_matrix_x.shape)
    train_matrix_y=np.array(train_df[0])
    rds_val_data = robjects.r['readRDS'](args.vali)  # Replace 'file.rds' with your file path
    val_df = robjects.conversion.rpy2py(rds_val_data)
    # val_matrix_x = np.transpose(np.array(val_df[1]), (0, 3, 1, 2))
    val_matrix_x = np.array(val_df[1])
    reshaped_data = val_matrix_x.reshape(val_matrix_x.shape[0], -1)
    normalized_data = (reshaped_data - reshaped_data.min(axis=0)) / (
            reshaped_data.max(axis=0) - reshaped_data.min(axis=0) + 1e-8)
    val_matrix_x = normalized_data.reshape(val_matrix_x.shape)
    val_matrix_y = np.array(val_df[0])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    train_lab = torch.tensor(train_matrix_y,dtype=torch.int64).to(device)
    vali_lab = torch.tensor(val_matrix_y,dtype=torch.int64).to(device)
    # train_lab_one_hot = torch.nn.functional.one_hot(train_lab, num_classes=2)
    # vali_lab_one_hot = torch.nn.functional.one_hot(vali_lab, num_classes=2)
    model = ResidualModule(5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
    num_epochs = args.epoch
    batch_size = args.batch
    train_matrix_x=torch.tensor(train_matrix_x,dtype=torch.float32).to(device)
    val_matrix_x=torch.tensor(val_matrix_x,dtype=torch.float32).to(device)
    train_dataset = TensorDataset(train_matrix_x, train_lab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_matrix_x, vali_lab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    best_val_accuracy = 0.0
    best_score = None
    loss_history = []
    val_loss_history = []
    val_score_history = []
    best_test_score = []
    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        count = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_accum += loss
            count += 1
        loss_history.append(float(loss_accum / count))
        # Validation
        model.eval()
        loss_accum = 0
        score_accum = 0
        count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                count += 1
                loss = criterion(outputs, labels)
                loss_accum += loss
                score_accum += accuracy(outputs, labels)
            val_loss_history.append(float(loss_accum / count))
            val_score_history.append(float(score_accum / count))
        # if best_score is None or best_score < np.mean(val_score_history[-1]):
        #     best_score = np.mean(val_score_history[-1])
        #     torch.save(model.state_dict(), os.path.join('data', model.__class__.__name__))  # save best model
        print(
            'Epoch #{}, train loss: {:.4f}, val loss: {:.4f}, val_accuracy: {:.4f}'.format(
                epoch,
                loss_history[-1],
                val_loss_history[-1],
                val_score_history[-1]
            ))
if __name__ == '__main__':
    main()