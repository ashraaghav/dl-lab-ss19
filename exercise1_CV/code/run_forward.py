import torch
import numpy as np
import matplotlib.pyplot as plt

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints


def normalize_keypoints(keypoints, img_shape):
    if img_shape[-1] != img_shape[-2]:
        raise ValueError("Only square images are supported")
    return keypoints/img_shape[-1]

# import os
# os.chdir('exercise1_CV/code')


def evaluate(model, loader, split='Test'):
    """
    Evaluation method
    :param model: ResNet Model to evaluate
    :param loader: data loader containing the images and their corresponding keypoints & weights
    :param split: Name of which dataset is used - Train or Test
    :return: accuracy on the data
    """
    scores = np.array([])
    with torch.no_grad():
        for i, (img, keypoints, weights) in enumerate(loader):
            img = img.to(cuda)
            keypoints = keypoints.to(cuda)
            weights = weights.to(cuda).float()
            # normalize keypoints to [0, 1] range
            keypoints = normalize_keypoints(keypoints, img.shape)

            # predict keypoints for the image
            outputs = model(img, '')
            # de-normalize keypoint coordinates
            outputs = outputs*img.shape[-1]
            keypoints = keypoints*img.shape[-1]

            # calculate mpjpe for each image
            score = mpjpe(outputs, keypoints, weights)
            scores = np.append(scores, score)

            if i > 10:
                break

        # average MPJPE on all images
        score = np.mean(scores)
        print('{0:>10} MPJPE of the model on the {1} images: {2:.4f}'.format(split, len(loader), score))
    return score


def mpjpe(preds, labels, weights):
    """
    Function that computes the MPJPE i.e., average euclidean distance between predicted & actual keypoints
    :param preds: predicted keypoint coordinates
    :param labels: annotated keypoint coordinates
    :param weights: weights of keypoints (1 if present, else 0)
    :return: MPJPE of each image as numpy array
    """
    # # weights adjusted for all points (both x and y coordinates)
    # w = torch.zeros([weights.shape[0], weights.shape[1] * 2]).to(cuda)
    # w[:, 0::2] = weights
    # w[:, 1::2] = weights

    # repeat weights for both x and y coordinates
    weights = weights.transpose(0, 1).repeat(1, 2).view(-1, weights.shape[0]).transpose(0, 1).float()

    # calculating eucledian distance between all keypoints (PJPE)
    dist = torch.nn.PairwiseDistance(2)
    score = dist(preds*weights, labels*weights)
    # find MPJPE for each image
    score = score / (torch.sum(weights, dim=1) / 2)

    return score.cpu().detach().numpy()


def weightedMSE(preds, labels, weights, train_criterion):
    """
    Function that computes the weighted MSE to use as training objective
    :param preds: predicted keypoint coordinates
    :param labels: annotated keypoint coordinates
    :param weights: weights of keypoints (1 if present, else 0)
    :return: weighted L2 loss
    """
    # # weights adjusted for all points (both x and y coordinates)
    # w = torch.zeros([weights.shape[0], weights.shape[1] * 2]).to(cuda)
    # w[:, 0::2] = weights
    # w[:, 1::2] = weights

    # repeat weights for both x and y coordinates
    weights = weights.transpose(0, 1).repeat(1, 2).view(-1, weights.shape[0]).transpose(0, 1)

    # calculating mean squared error between all keypoints
    loss = torch.pow(preds*weights - labels*weights, 2)
    loss = torch.sum(loss, dim=1) / torch.sum(weights, dim=1)
    return loss.mean()


def train(model, train_loader, valid_loader, train_criterion, epochs=10, valid_split=0.8):
    """
    Function to train the given ResNet18 model
    :param model: ResNet model to train
    :param loader: data loader containing the images and their corresponding keypoints & weights
    :param train_criterion: function that provides the objective to optimize in training
    :param epochs: number of epochs to train
    :param valid_split: percentage split between training & evaluation dataset
    :return: model and the training statistics
    """

    # # split training dataset into train & validation
    # train_samples = int(len(loader)*valid_split)
    # valid_samples = len(loader)-train_samples
    # train, valid = torch.utils.data.random_split(loader, [train_samples, valid_samples])

    # initialize optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_criterion = train_criterion()

    train_losses, train_scores, valid_scores = [], [], []
    total_steps = len(train_loader)
    # run training for given number of epochs
    for e in range(epochs):
        print('#' * 120)
        for b, (img, keypoints, weights) in enumerate(train_loader):
            img = img.to(cuda)
            keypoints = keypoints.to(cuda)
            weights = weights.to(cuda).float()
            # normalize keypoints to [0, 1] range
            keypoints = normalize_keypoints(keypoints, img.shape)

            # forward pass
            preds = model(img, '')
            # compute loss
            loss = weightedMSE(preds, keypoints, weights, train_criterion)
            # backward pass
            optimizer.zero_grad()  # zero out gradients for new minibatch
            loss.backward()
            optimizer.step()

            # collect stats about training
            train_losses.append(loss.item())

            if (b+1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(
                    e+1, epochs, b+1, total_steps, loss.item()))

            if b > 10:
                break

        print("Evaluating after epoch....")
        # evaluate on training & validation per epoch
        # tr_mpjpe = evaluate(model, train_loader, split='Train')
        # train_scores.append(tr_mpjpe)
        va_mpjpe = evaluate(model, valid_loader, split='Test')
        valid_scores.append(va_mpjpe)

    # combining all stats into one dict
    stats = {'train_loss': train_losses, 'train_score': train_scores, 'valid_score': valid_scores}

    return model, stats


if __name__ == '__main__':
    PATH_TO_CKPT = './trained_net.model'

    # create device and model
    cuda = torch.device('cuda')
    model = ResNetModel(pretrained=True)
    model.load_state_dict(torch.load(PATH_TO_CKPT))
    model.to(cuda)

    train_loader = get_data_loader(batch_size=32, is_train=True)
    valid_loader = get_data_loader(batch_size=1, is_train=False)

    # TODO TASK 1: Training - define loss, optimizer, intermediate snapshots
    # handle missing keypoints - squared L2 loss
    print('TASK 1:')
    print('-' * 40)
    print('Training ResNet with MSE loss')
    model, stats = train(model, train_loader, valid_loader, train_criterion=torch.nn.MSELoss)

    print('Training complete !!')
    print('-' * 40)

    for idx, (img, keypoints, weights) in enumerate(valid_loader):
        img = img.to(cuda)
        keypoints = keypoints.to(cuda)
        weights = weights.to(cuda)

        # normalize keypoints to [0, 1] range
        keypoints = normalize_keypoints(keypoints, img.shape)

        # apply model
        pred = model(img, '')

        # show results
        img_np = np.transpose(img.cpu().detach().numpy(), [0, 2, 3, 1])
        img_np = np.round((img_np + 1.0) * 127.5).astype(np.uint8)
        kp_pred = pred.cpu().detach().numpy().reshape([-1, 17, 2])
        kp_gt = keypoints.cpu().detach().numpy().reshape([-1, 17, 2])
        vis = weights.cpu().detach().numpy().reshape([-1, 17])

        for bid in range(img_np.shape[0]):
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(img_np[bid]), ax1.axis('off'), ax1.set_title('input + gt')
            plot_keypoints(ax1, kp_gt[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            ax2.imshow(img_np[bid]), ax2.axis('off'), ax2.set_title('input + pred')
            plot_keypoints(ax2, kp_pred[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            plt.show()
