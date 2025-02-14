import os
import time
import json
import torch
import torch.nn as nn

from torch import optim
from model import VQAModel
from build_dataset import data_loader
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import classification_report

DATA_DIR = os.getenv("PREPROCESSED_DIR")
CKPT_DIR = os.getenv("CHECKPOINT_DIR")
RES_DIR = os.getenv("RESULT_DIR")
LOG_DIR = os.getenv("LOG_DIR")

BLINDED = True
BATCH_SIZE = 16
MAX_QU_LEN = 60
NUM_WORKER = 8
FEATURE_SIZE, WORD_EMBED = 1024, 300
NUM_HIDDEN, HIDDEN_SIZE = 2, 512
LEARNING_RATE, STEP_SIZE, GAMMA = 0.001, 10, 0.1
EPOCH = 30

device = torch.device(os.getenv('DEVICE'))


def train():
    dataloader = data_loader(input_dir=DATA_DIR, batch_size=BATCH_SIZE, max_qu_len=MAX_QU_LEN, num_worker=NUM_WORKER)
    qu_vocab_size = dataloader['train'].dataset.qu_vocab.vocab_size
    ans_vocab_size = dataloader["train"].dataset.ans_vocab.vocab_size

    model = VQAModel(feature_size=FEATURE_SIZE, qu_vocab_size=qu_vocab_size, ans_vocab_size=ans_vocab_size,
                     word_embed=WORD_EMBED, hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN, blinded=BLINDED).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()

    print('>> start training')
    start_time = time.time()
    for epoch in tqdm(range(EPOCH)):
        epoch_loss = {key: 0 for key in ['train', 'val']}

        model.train()
        for idx, sample in tqdm(enumerate(dataloader['train'])):
            image = sample['image'].to(device=device)
            question = sample['question'].to(device=device)
            label = sample['answer'].to(device=device)
            # forward
            logits = model(image, question)
            loss = criterion(logits, label)
            epoch_loss['train'] += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for idx, sample in tqdm(enumerate(dataloader['val'])):
            image = sample['image'].to(device=device)
            question = sample['question'].to(device=device)
            label = sample['answer'].to(device=device)
            with torch.no_grad():
                logits = model(image, question)
                loss = criterion(logits, label)
            epoch_loss['val'] += loss.item()

        # statistic
        for phase in ['train', 'val']:
            epoch_loss[phase] /= len(dataloader[phase])
            with open(os.path.join(LOG_DIR, f'{phase}_log.txt'), 'a') as f:
                f.write(str(epoch + 1) + '\t' + str(epoch_loss[phase]) + '\n')
        print('Epoch:{}/{} | Training Loss: {train:6f} | Validation Loss: {val:6f}'.format(epoch + 1, EPOCH,
                                                                                           **epoch_loss))

        scheduler.step()
        early_stop = early_stopping(model, epoch_loss['val'])
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'model-epoch-{epoch + 1}.pth'))
        if early_stop:
            print(f'>> Early stop at {epoch + 1} epoch')
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f">> Finishing training | Training Time:{training_time // 60:.0f}m:{training_time % 60:.0f}s")

    model.eval()
    test_loss = 0
    all_pred_answers = []
    all_ground_answers = []
    for idx, sample in tqdm(enumerate(dataloader['test'])):
        image = sample['image'].to(device=device)
        question = sample['question'].to(device=device)
        label = sample['answer'].to(device=device)
        with torch.no_grad():
            logits = model(image, question)
            test_loss += criterion(logits, label)
            print('Test Loss:', test_loss / len(dataloader['test']))
            all_pred_answers.extend(logits.argmax(-1).detach().cpu().numpy().tolist())
            all_ground_answers.extend(label.detach().cpu().numpy().tolist())
        report = classification_report(all_pred_answers, all_ground_answers, output_dict=True)
        print(json.dumps(report, indent=4))
        with open('ground_and_pred_answers.json', 'w') as f:
            json.dump({'ground_answers': all_ground_answers, 'pred_answers': all_pred_answers}, f, indent=4)


def early_stopping(model, epoch_loss, patience=7):
    early_stop = False
    if not bool(early_stopping.__dict__):
        early_stopping.best_loss = epoch_loss
        early_stopping.record_loss = epoch_loss
        early_stopping.counter = 0

    if epoch_loss < early_stopping.best_loss:
        early_stopping.best = epoch_loss
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'best_model.pth'))

    if epoch_loss > early_stopping.record_loss:
        early_stopping.counter += 1
        if early_stopping.counter >= patience:
            early_stop = True
    else:
        early_stopping.counter = 0
        early_stopping.record_loss = epoch_loss

    return early_stop


if __name__ == '__main__':

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)
    train()
