import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

from model.data import get_data_loader
from model.model import ResNetModelR, ResNetModelS
from utils.plot_util import plot_keypoints


def normalize_keypoints(keypoints, img_shape):
    if img_shape[-1] != img_shape[-2]:
        raise ValueError("Only square images are supported")
    return keypoints / img_shape[-1]


def evaluate(model, loader, split='Test'):
    """
    Evaluation method
    :param model: ResNet Model to evaluate
    :param loader: data loader containing the images and their corresponding keypoints & weights
    :param split: Name of which dataset is used - Train or Test
    :return: accuracy on the data
    """
    scores = np.array([])
    dist = torch.nn.PairwiseDistance(2)

    with torch.no_grad():
        for i, (img, keypoints, weights) in enumerate(loader):
            img = img.to(cuda)
            keypoints = keypoints.to(cuda)
            weights = weights.to(cuda).float()
            # normalize keypoints to [0, 1] range
            keypoints = normalize_keypoints(keypoints, img.shape)

            # predict keypoints for the image
            outputs = model(img)
            # de-normalize keypoint coordinates
            outputs = outputs * img.shape[-1]
            keypoints = keypoints * img.shape[-1]

            # calculate mpjpe for each image
            score = mpjpe(outputs, keypoints, weights, dist)
            scores = np.append(scores, score)

        # average MPJPE on all images
        score = np.mean(scores)
        print('{0:>10} MPJPE of the model on the {1} images: {2:.4f}'.format(split, len(loader.dataset), score))
    return score


def mpjpe(preds, labels, weights, dist):
    """
    Function that computes the MPJPE i.e., average euclidean distance between predicted & actual keypoints
    :param preds: predicted keypoint coordinates of shape n*2k
    :param labels: annotated keypoint coordinates of shape n*2k
    :param weights: weights of keypoints of shape n*k (1 if present, else 0)
    :return: MPJPE of each image as numpy array (n-length)
    """
    # repeat weights for both x and y coordinates
    weights = weights.transpose(0, 1).repeat(1, 2).view(-1, weights.shape[0]).transpose(0, 1).float()
    preds = preds * weights
    labels = labels * weights
    # reshaping to B*K*2
    labels = labels.view(preds.shape[0], int(preds.shape[1]/2), 2).transpose(1, 2)
    preds = preds.view(preds.shape[0], int(preds.shape[1]/2), 2).transpose(1, 2)

    # calculating eucledian distance between all keypoints (PJPE)
    score = dist(preds, labels)
    # # find MPJPE for each image
    score = torch.sum(score, dim=1) / (torch.sum(weights, dim=1) / 2)

    return score.cpu().detach().numpy()


def weightedL2Loss(preds, labels, weights):
    """
    Function that computes the weighted L2 loss to use as training objective
    :param preds: predicted keypoint coordinates
    :param labels: annotated keypoint coordinates
    :param weights: weights of keypoints (1 if present, else 0)
    :return: weighted L2 loss
    """
    # repeat weights for both x and y coordinates
    weights = weights.transpose(0, 1).repeat(1, 2).view(-1, weights.shape[0]).transpose(0, 1)

    # average weighted squared sum over the squared distance
    loss = torch.sum(torch.pow(preds*weights - labels*weights, 2), dim=1)
    loss = loss / (torch.sum(weights, dim=1) / 2)
    loss = torch.mean(loss)
    return loss


def train(model, train_loader, valid_loader, epochs=10, valid_split=0.8, initial_eval=False, name='t'):
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

    train_losses, train_scores, valid_scores = [], [], []

    # initialize optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    dist = torch.nn.PairwiseDistance(2)

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
            preds = model(img)
            # compute loss
            loss = weightedL2Loss(preds, keypoints, weights)
            # backward pass
            optimizer.zero_grad()  # zero out gradients for new minibatch
            loss.backward()
            optimizer.step()

            # collect stats about training
            train_losses.append(loss.item())

            if (b + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(
                    e + 1, epochs, b + 1, total_steps, loss.item()))

        print("Evaluating after epoch....")
        # evaluate on training & validation per epoch
        tr_mpjpe = evaluate(model, train_loader, split='Train')
        train_scores.append(tr_mpjpe)
        va_mpjpe = evaluate(model, valid_loader, split='Test')
        valid_scores.append(va_mpjpe)

    # combining all stats into one dict
    stats = {'train_loss': train_losses, 'train_score': train_scores, 'valid_score': valid_scores}

    # save model & stats about model after training
    save_model_str = SAVE_PATH
    save_id = time.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(save_model_str):
        os.makedirs(save_model_str, exist_ok=True)
    save_model_str += name+'_'+save_id+'_ckpt.model'
    torch.save(model.state_dict(), save_model_str)
    save_stats_str = SAVE_PATH+name+'_'+save_id+'_stat.json'
    with open(save_stats_str, 'w') as f:
        f.write(json.dumps(stats))

    return model, stats


if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('DL Lab - Exercise 1')
    cmdline_parser.add_argument('--train',
                                help='To run training for all models',
                                action='store_true')
    cmdline_parser.add_argument('--validate',
                                help='To run validation for the given 2 models. Note: If validating, specify model files in main',
                                action='store_true')
    args, _ = cmdline_parser.parse_known_args()

    if not args.train and not args.validate:
        print('Provide command line arguments. Run with -h for details')

    if args.train:
        PATH_TO_CKPT = '../trained_net.model'
        SAVE_PATH = '../saved_models/'
        
        # initialize device
        cuda = torch.device('cuda')
        # loading dataset
        valid_loader = get_data_loader(batch_size=16, is_train=False)
        train_loader = get_data_loader(batch_size=16, is_train=True)

        print('TASK 1.1: REGRESSION - (No pretraining)')
        print('-' * 40)
        print("Initializing model...")
        # create model
        model = ResNetModelR(pretrained=False)
        # model.load_state_dict(torch.load(PATH_TO_CKPT))
        model.to(cuda)
        print('Training untrained ResNet with L2 loss')
        model, stats = train(model, train_loader, valid_loader, epochs=20, name='t1_no')
        print('Training complete !!')
        print('=' * 120)

        print('TASK 1.2: REGRESSION - (pretraining)')
        print('-' * 40)
        print("Initializing model...")
        # create model
        model = ResNetModelR(pretrained=True)
        # model.load_state_dict(torch.load(PATH_TO_CKPT))
        model.to(cuda)
        print('Training pretrained ResNet with L2 loss')
        model, stats = train(model, train_loader, valid_loader, epochs=20, name='t1_pr')
        print('Training complete !!')
        print('=' * 120)

        print('TASK 2: SOFTARGMAX')
        print('-' * 40)
        print("Initializing model...")
        # create model
        model = ResNetModelS(pretrained=True)
        # model.load_state_dict(torch.load(PATH_TO_CKPT))
        model.to(cuda)
        print('Training pretrained ResNet with L2 loss')
        model, stats = train(model, train_loader, valid_loader, epochs=20, name='t2_pr')
        print('Training complete !!')
        print('=' * 120)

    if args.validate:
        print('VALIDATING....')
        # Load saved models
        LOAD_PATH_S = '../output/t2_pr_2019-05-05_22-04-27_ckpt.model'
        LOAD_PATH_R = '../output/t1_pr_2019-05-04_11-32-59_ckpt.model'
        # LOAD_PATH = PATH_TO_CKPT
        
        # create device and model
        cuda = torch.device('cuda')
        modelS = ResNetModelS(pretrained=True)
        modelS.load_state_dict(torch.load(LOAD_PATH_S))
        modelS.to(cuda)
        modelR = ResNetModelR(pretrained=True)
        modelR.load_state_dict(torch.load(LOAD_PATH_R))
        modelR.to(cuda)
        
        dist = torch.nn.PairwiseDistance(2)
        
        # Validate on all test images, one by one
        valid_loader = get_data_loader(batch_size=1, is_train=False)
        for idx, (img, keypoints, weights) in enumerate(valid_loader):
            img = img.to(cuda)
            keypoints = keypoints.to(cuda)
            weights = weights.to(cuda).float()
        
            # normalize keypoints to [0, 1] range
            keypoints = normalize_keypoints(keypoints, img.shape)
        
            # apply model
            predR = modelR(img)
            predS = modelS(img)
        
            # calculate mpjpe for each image
            scoreR = mpjpe(predR*img.shape[-1], keypoints*img.shape[-1], weights, dist)[0]
            scoreR = str(np.round(scoreR, 2))
            scoreS = mpjpe(predS*img.shape[-1], keypoints*img.shape[-1], weights, dist)[0]
            scoreS = str(np.round(scoreS, 2))
            print("MPJPE  for image {}: regression {}, softargmax {}".format(idx, scoreR, scoreS))
        
            # show results
            img_np = np.transpose(img.cpu().detach().numpy(), [0, 2, 3, 1])
            img_np = np.round((img_np + 1.0) * 127.5).astype(np.uint8)
            kp_pred_R = predR.cpu().detach().numpy().reshape([-1, 17, 2])
            kp_pred_S = predS.cpu().detach().numpy().reshape([-1, 17, 2])
            kp_gt = keypoints.cpu().detach().numpy().reshape([-1, 17, 2])
            vis = weights.cpu().detach().numpy().reshape([-1, 17])
        
            for bid in range(img_np.shape[0]):
                fig = plt.figure()
                ax1 = fig.add_subplot(131)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)
                ax1.imshow(img_np[bid]), ax1.axis('off'), ax1.set_title('ground truth', fontdict={'fontsize':10})
                plot_keypoints(ax1, kp_gt[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
                ax2.imshow(img_np[bid]), ax2.axis('off'), ax2.set_title('regression \n (MPJPE:'+scoreR+')', fontdict={'fontsize':10})
                plot_keypoints(ax2, kp_pred_R[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
                ax3.imshow(img_np[bid]), ax3.axis('off'), ax3.set_title('softargmax \n (MPJPE:'+scoreS+')', fontdict={'fontsize':10})
                plot_keypoints(ax3, kp_pred_S[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
                # plt.figtext(0.5, 0.1, "MPJPE: "+str(score), wrap=True, horizontalalignment='center', fontsize=12)
                plt.savefig('example.png', bbox_inches='tight')
                plt.show()
