
import os, json, random
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
from tqdm import tqdm

pinyin_list = ['a', 'ai', 'an', 'ang', 'ao', 'ba', 'bai', 'ban', 'bang', 'bao', 'be', 'bei', 'ben', 'beng', 'bi', 'bian', 'bianr', 'biao', 'bie', 'bin', 'bing', 'bo', 'bu', 'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng', 'cha', 'chai', 'chan', 'chang', 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou', 'chu', 'chua', 'chuai', 'chuan', 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong', 'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo', 'da', 'dai', 'dan', 'dang', 'dao', 'de', 'dei', 'den', 'deng', 'di', 'dia', 'dian', 'diao', 'die', 'ding', 'diu', 'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo', 'e', 'ei', 'en', 'ng', 'eng', 'er', 'fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu', 'ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng', 'gong', 'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo', 'ha', 'hai', 'hair', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hong', 'hou', 'hu', 'hua', 'huai', 'huan', 'huang', 'hui', 'huir', 'hun', 'huo', 'ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong', 'jiu', 'ju', 'juan', 'jue', 'jun', 'ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'kei', 'ken', 'keng', 'kong', 'kou', 'ku', 'kua', 'kuai', 'kuan', 'kuang', 'kui', 'kun', 'kuo', 'la', 'lai', 'lan', 'lang', 'lao', 'le', 'lei', 'leng', 'li', 'lia', 'lian', 'liang', 'liao', 'lie', 'lin', 'ling', 'liu', 'lo', 'long', 'lou', 'lu', 'lv', 'luan', 'lve', 'lue', 'lun', 'luo', 'ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi', 'mian', 'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu', 'na', 'nar', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ni', 'nia', 'nian', 'niang', 'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nou', 'nu', 'nv', 'nuan', 'nve', 'nue', 'nuo', 'o', 'ou', 'pa', 'pai', 'pan', 'pang', 'pao', 'pe', 'pei', 'pen', 'peng', 'pi', 'pian', 'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu', 'qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong', 'qiu', 'qu', 'quan', 'que', 'qun', 'ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri', 'rong', 'rou', 'ru', 'rua', 'ruan', 'rui', 'run', 'ruo', 'sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai', 'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shir', 'shou', 'shu', 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'si', 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo', 'ta', 'tai', 'tan', 'tang', 'tao', 'te', 'tei', 'teng', 'ti', 'tian', 'tianr', 'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo', 'wa', 'wai', 'wan', 'wanr', 'wang', 'wei', 'weir', 'wen', 'weng', 'wo', 'wu', 'xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing', 'xiong', 'xiu', 'xu', 'xuan', 'xue', 'xun', 'ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying', 'yo', 'yong', 'you', 'yu', 'yuan', 'yue', 'yun', 'za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha', 'zhai', 'zhan', 'zhang', 'zhao', 'zhe', 'zhei', 'zhen', 'zheng', 'zher', 'zhi', 'zhong', 'zhou', 'zhu', 'zhua', 'zhuai', 'zhuan', 'zhuang', 'zhui', 'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu', 'zuan', 'zui', 'zun', 'zuo', 'gair', 'yir', 'jinr', 'menr', 'her', 'gunr', 'huanr', 'jiar', 'duir', 'dir', 'banr', 'yanr', 'cir', 'genr', 'ger', 'tongr', 'renr', 'kuair', 'yuanr', 'liar', 'dunr', 'fur', 'zaor', 'yangr', 'ter', 'yingr', 'fanr', 'tuor', 'pir', 'jingr', 'hanr', 'duor', 'tanr', 'zhunr', 'jir', 'huor', 'far', 'dianr', 'zhur', 'tour', 'qir', 'mingr', 'lanr', 'lir', 'daor']
# tones = [1, 2, 3, 4, 5, 6]

pinyin_to_label = {p: i for i, p in enumerate(pinyin_list)}
label_to_pinyin = {i: p for i, p in enumerate(pinyin_list)}


from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('TencentGameMate/chinese-hubert-base')
hubert_model = HubertModel.from_pretrained('TencentGameMate/chinese-hubert-base')
hubert_model = hubert_model.to('cuda')

biaobei_py_list = json.loads(open('biaobei_py_list.json', "r").read())

audio_py_pairs = [['lab_wav_pairs/' + p['file_name'] + '.wav', pc[0], pc[1], pc[2]] for p in biaobei_py_list[:8000] for pc in p['pinyin_clips']]
# random.shuffle(audio_py_pairs)

py_count = {}

for ap in audio_py_pairs:
    py = ap[1][:-1]
    if py not in py_count:
        py_count[py] = 1
    else:
        py_count[py] += 1
        
aug_audio_py_pairs = []

for ap in audio_py_pairs:
    # print(ap)
    py = ap[1][:-1]
    aug_audio_py_pairs.append(ap + [0])
    if py_count[py] < 30:
        for i in range(30 // py_count[py]):
            aug_audio_py_pairs.append(ap + [1])

f_audio_py_pairs = []

feature_dir = 'extracted_features_thubert_aug/'
if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)

for i, pair in enumerate(tqdm(aug_audio_py_pairs)):
    file_path, py, start, end, aug = pair
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file_path, int(start * 48000), int((end-start) * 48000))
    SPEECH_WAVEFORM = torchaudio.functional.resample(SPEECH_WAVEFORM, SAMPLE_RATE, 16000)
    # print(SPEECH_WAVEFORM.shape)
    
    if aug == 1:
        effects = []
        if random.random() < 0.2:
            effects.append(['gain', '-n'])
        if random.random() < 0.5:
            pitch = random.randint(-10, 10)
            effects.append(['pitch', f'{pitch}'])
        if random.random() < 0.6:
            speed = 0.5 + random.random()            
            effects.append(['speed', f'{speed:.5f}'])
        if random.random() > 0.5:
            freq = random.randint(2000, 4000)
            effects.append(["lowpass", f'{freq}'])
        effects.append(['rate', f'{SAMPLE_RATE}'])
        # print(effects)
        SPEECH_WAVEFORM, _ = torchaudio.sox_effects.apply_effects_tensor(SPEECH_WAVEFORM, SAMPLE_RATE, effects)
        
    SPEECH_WAVEFORM = F.pad(SPEECH_WAVEFORM, (0, 16000-SPEECH_WAVEFORM.shape[1]), mode='constant', value=0)
    input_values = feature_extractor(SPEECH_WAVEFORM[0], return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to('cuda')

    with torch.no_grad():
        outputs = hubert_model(input_values)
        last_hidden_state = outputs.last_hidden_state
    
    f_npy = feature_dir + str(i) + '.npy'
    with open(f_npy, 'wb') as f:
        np.save(f, last_hidden_state.cpu().detach().numpy())
    
    f_audio_py_pairs.append((file_path, py, start, end, f_npy))
    # print(stacked, py)
    
json_object = json.dumps(f_audio_py_pairs, indent=4)

with open("f_audio_py_pairs_thubert_aug.json", "w") as outfile:
    outfile.write(json_object)
    