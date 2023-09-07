
import os, json, random
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime

pinyin_list = ['a', 'ai', 'an', 'ang', 'ao', 'ba', 'bai', 'ban', 'bang', 'bao', 'be', 'bei', 'ben', 'beng', 'bi', 'bian', 'bianr', 'biao', 'bie', 'bin', 'bing', 'bo', 'bu', 'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng', 'cha', 'chai', 'chan', 'chang', 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou', 'chu', 'chua', 'chuai', 'chuan', 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong', 'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo', 'da', 'dai', 'dan', 'dang', 'dao', 'de', 'dei', 'den', 'deng', 'di', 'dia', 'dian', 'diao', 'die', 'ding', 'diu', 'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo', 'e', 'ei', 'en', 'ng', 'eng', 'er', 'fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu', 'ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng', 'gong', 'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo', 'ha', 'hai', 'hair', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hong', 'hou', 'hu', 'hua', 'huai', 'huan', 'huang', 'hui', 'huir', 'hun', 'huo', 'ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong', 'jiu', 'ju', 'juan', 'jue', 'jun', 'ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'kei', 'ken', 'keng', 'kong', 'kou', 'ku', 'kua', 'kuai', 'kuan', 'kuang', 'kui', 'kun', 'kuo', 'la', 'lai', 'lan', 'lang', 'lao', 'le', 'lei', 'leng', 'li', 'lia', 'lian', 'liang', 'liao', 'lie', 'lin', 'ling', 'liu', 'lo', 'long', 'lou', 'lu', 'lv', 'luan', 'lve', 'lue', 'lun', 'luo', 'ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi', 'mian', 'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu', 'na', 'nar', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ni', 'nia', 'nian', 'niang', 'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nou', 'nu', 'nv', 'nuan', 'nve', 'nue', 'nuo', 'o', 'ou', 'pa', 'pai', 'pan', 'pang', 'pao', 'pe', 'pei', 'pen', 'peng', 'pi', 'pian', 'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu', 'qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong', 'qiu', 'qu', 'quan', 'que', 'qun', 'ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri', 'rong', 'rou', 'ru', 'rua', 'ruan', 'rui', 'run', 'ruo', 'sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai', 'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shir', 'shou', 'shu', 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'si', 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo', 'ta', 'tai', 'tan', 'tang', 'tao', 'te', 'tei', 'teng', 'ti', 'tian', 'tianr', 'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo', 'wa', 'wai', 'wan', 'wanr', 'wang', 'wei', 'weir', 'wen', 'weng', 'wo', 'wu', 'xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing', 'xiong', 'xiu', 'xu', 'xuan', 'xue', 'xun', 'ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying', 'yo', 'yong', 'you', 'yu', 'yuan', 'yue', 'yun', 'za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha', 'zhai', 'zhan', 'zhang', 'zhao', 'zhe', 'zhei', 'zhen', 'zheng', 'zher', 'zhi', 'zhong', 'zhou', 'zhu', 'zhua', 'zhuai', 'zhuan', 'zhuang', 'zhui', 'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu', 'zuan', 'zui', 'zun', 'zuo', 'gair', 'yir', 'jinr', 'menr', 'her', 'gunr', 'huanr', 'jiar', 'duir', 'dir', 'banr', 'yanr', 'cir', 'genr', 'ger', 'tongr', 'renr', 'kuair', 'yuanr', 'liar', 'dunr', 'fur', 'zaor', 'yangr', 'ter', 'yingr', 'fanr', 'tuor', 'pir', 'jingr', 'hanr', 'duor', 'tanr', 'zhunr', 'jir', 'huor', 'far', 'dianr', 'zhur', 'tour', 'qir', 'mingr', 'lanr', 'lir', 'daor', 'niur']
# tones = [1, 2, 3, 4, 5, 6]

pinyin_to_label = {p: i for i, p in enumerate(pinyin_list)}
label_to_pinyin = {i: p for i, p in enumerate(pinyin_list)}

class PinyinDataset(Dataset):
    def __init__(self, audio_py_pairs):
        self.audio_py_pairs = audio_py_pairs
        
    def __len__(self):
        return len(self.audio_py_pairs)
    
    # def get_spec_py_pair(self, pair):
    #     file_path, py, start, end = pair
        
    #     SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file_path, int(start * 24000), int((end-start) * 24000))

    #     n_fft = 1024
    #     win_length = None
    #     hop_length = 512
    #     n_mels = 224

    #     # define transformation
    #     melspectrogram = T.MelSpectrogram(
    #         n_fft=n_fft,
    #         win_length=win_length,
    #         hop_length=hop_length,
    #         n_mels=n_mels,
    #         center=True,
    #         pad_mode="reflect",
    #         power=2.0,
    #     )
    #     # Perform transformation
    #     try:
    #         spec = melspectrogram(SPEECH_WAVEFORM)
    #     except Exception as e:
    #         print('!!!', file_path, int(start * 24000), int((end-start) * 24000), e)
    #     if spec.shape[2] < 33:
    #         spec= F.pad(spec, (0, 33 - spec.shape[2]))
    #     # else:
    #     #     spec = spec[:, :28]
        
    #     py_wo_tone = py[:-1]
    #     label = pinyin_to_label[py_wo_tone]

    #     return spec, label
    
    def get_audio_py_pair(self, pair):
        file_path, py, start, end, f_npy = pair
        
        stacked = torch.from_numpy(np.load(open(f_npy, 'rb')))
        
        py_wo_tone = py[:-1]
        label = pinyin_to_label[py_wo_tone]
        
        # sp = torch.split(stacked, [1, 12])
        
        return stacked, label, file_path, start, end, py
        
    def __getitem__(self, idx):
        return self.get_audio_py_pair(self.audio_py_pairs[idx])
    
    
audio_py_pairs = json.loads(open('f_audio_py_pairs_thubert_aug.json', "r").read())
random.seed(1)
random.shuffle(audio_py_pairs)

train_pairs, test_pairs = audio_py_pairs[:int(len(audio_py_pairs)*.9)], audio_py_pairs[int(len(audio_py_pairs)*.9):]

train_data = PinyinDataset(train_pairs)
test_data = PinyinDataset(test_pairs)

tr = set([tp[1][:-1] for tp in train_pairs])
te = set([tp[1][:-1] for tp in test_pairs])
print(len(tr), len(te), len(audio_py_pairs))
te - tr

train_loader = DataLoader(dataset=train_data, num_workers=8, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_data, num_workers=8, batch_size=128, shuffle=False)


def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, len(pinyin_list))
        
    return model

model = models.resnet18(pretrained=False)
change_layers(model)

from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

# model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()
model_dir = './checkpoints/' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

for epoch in range(1, 51):
    model.train()
    running_loss = 0.0
    correct = 0
    total_len = 0
    for i, data in enumerate(train_loader):
        inputs, labels, _, _, _, _ = data
        # inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        correct += (predicted == labels).float().sum()
        total_len += len(labels)
    accuracy = 100 * correct / total_len
    print(f'Epoch [{epoch}/200], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}%')
    
    if epoch % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_dir, 'ckp_' + str(epoch) + '.pt'))
        
    model.eval()
    correct = 0
    total = 0
    for inputs, labels, file_path, start, end, py in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        with torch.no_grad():
            output = model(inputs)
        _, predicted = torch.max(output,1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

    print('Accuracy of the model: %.3f %%' %((100*correct)/total))
    
    scheduler.step()
