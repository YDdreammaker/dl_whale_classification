# 🥈HappyWhale - Dolphin and Whale

<div style="text-align:center"><img src=./img/main.png?raw=true /></div>
<br>

# Contents

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🧐Task Description](https://github.com/YDdreammaker/dl_whale_classification#task-description-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🏆Project Result](https://github.com/YDdreammaker/dl_whale_classification#project-result-1)**

<!-- #### &nbsp;&nbsp;&nbsp;&nbsp;**[⚙Installation](https://github.com/YDdreammaker/dl_whale_classification#installation-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🕹Command Line Interface](https://github.com/YDdreammaker/dl_whale_classification#command-line-interface-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🤝Collaboration Tools](https://github.com/YDdreammaker/dl_whale_classification#collaboration-tools-1)** -->

#### &nbsp;&nbsp;&nbsp;&nbsp;**[👩‍👦‍👦Who Are We?](https://github.com/YDdreammaker/dl_whale_classification#who-are-we-1)**
<br>

# Task Description

### Subject https://www.kaggle.com/competitions/happy-whale-and-dolphin 
<br>
이번 대회의 주제는 돌고래와 고래의 사진으로 individual_id 분류하는 문제였습니다.  돌고래와 고래의 지느러미에 사람의 지문과 같이 각 개체를 분류할 수 있는 특징이 있다고 생각해 해당 문제를 object recognition 이론으로 접근하였습니다.

<br>
돌고래의 사진을 지느러미와 지느러미를 포함한 몸통으로 object detection 진행하고, 해당 이미지를 가지고 각 개체를 분류하는 모델을 생성하였습니다.
<br>

### Data

- 훈련 데이터 : 51033장의 이미지와 해당 이미지의 종과 individual_id

- 테스트 데이터 : 27916장의 이미지
<br>

### Metric

<img src="./img/metric.png" />
<br>

| true  | predicted   | score |
|:-:|:-:|:-:|
| [x]  | [x, ?, ?, ?, ?]   | $$1\over1$$  |
| [x]  | [?, x, ?, ?, ?]   | $$1\over2$$  |
| [x]  | [?, ?, x, ?, ?]   | $$1\over3$$  |
| [x]  | [?, ?, ?, x, ?]   | $$1\over4$$  |
| [x]  | [?, ?, ?, ?, x]   | $$1\over5$$  |
<br>

# Project Result

* 은메달 47 등 / 1,613 팀

* Public LB Score: 0.85147 / Private LB Score: 0.81686

* 솔루션은 [이곳](https://www.notion.so/Solution-3ccc8fd2a39841a78d5726946b109707)에서 확인하실 수 있습니다.
<br>

<!-- # Installation

```shell
# clone repository
git clone https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you.git

# install necessary tools
pip install -r requirements.txt
```

### Dataset Structure

```shell
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...     
    └── *.jpg
```

### Code Structure

```shell
[code]
├── configs/ # configuration files
├── data_tools/ # modules for dataset
├── networks/ # modules for model architecture
├── postprocessing/ # modules for postprocessing during inference
├── schedulers/ # scheduler for learning rate, teacher forcing ratio
├── utils/ # useful utilities
├── inference_modules/ # modules for inference
├── train_modules/ # modules for train
├── README.md
├── requirements.txt
├── train.py
└── inference.py
```



# Command Line Interface

## Train

#### Train with single optimizer

```shell
$ python train.py --train_type single_opt --config_file './configs/EfficientSATRN.yaml'
```

#### Train with two optimizers for encoder and decoder

```shell
$ python train.py --train_type dual_opt --config_file './configs/EfficientSATRN.yaml'
```

#### Knowledge distillation training

```shell
$ python train.py --train_type distillation --config_file './configs/LiteSATRN.yaml' --teacher_ckpt 'TEACHER-MODEL_CKPT_PATH'
```

#### Train with Weight & Bias logging tool

```shell
$ python train.py --train_type single_opt --project_name <PROJECTNAME> --exp_name <EXPNAME> --config_file './configs/EfficientSATRN.yaml'
```

#### Arguments

##### `train_type (str)`: 학습 방식

* `'single_opt'`: 단일 optimizer를 활용한 학습을 진행합니다.
* `'dual_opt'`: 인코더, 디코더에 optimizer가 개별 부여된 학습을 진행합니다.
* `'distillation'`: Knowledge Distillation 학습을 진행합니다.

##### `config_file (str)`: 학습 모델의 configuration 파일 경로

- 모델 configuration은 아키텍처별로 상이하며, [이곳](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/configs/EfficientASTER.yaml)에서 해당 예시를 보실 수 있습니다.
- 학습 가능한 모델은 ***[EfficientSATRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientSATRN.py#L664)***, ***[EfficientASTER](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientASTER.py#L333)***, ***[SwinTRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/SWIN.py#L1023)***,    ***[LiteSATRN](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/3ffa06229659505fc2b4ef2ec652168b4ff7857b/networks/LiteSATRN.py#L548)*** 입니다.

##### `teacher_ckpt (str)`: Knowledge Distillation 학습 시 불러올 Teacher 모델 checkpoint 경로

##### `project_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 프로젝트명

##### `exp_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 실험명

---

## Inference

#### Inference with single model

```shell
$ python inference.py --inference_type single --checkpoint <MODELPATH.pth>
```

#### Ensemble inference

```shell
$ python inference.py --inference_type ensemble --checkpoint <MODEL1PATH.pth> <MODEL2PATH.pth> ...
```

#### Arguments

##### `inference_type (str)`: 추론 방식

- `single`: 단일 모델을 불러와 추론을 진행합니다.
- `ensemble`: 여러 모델을 불러와 앙상블 추론을 진행합니다.

##### `checkpoint (str)`: 불러올 모델 경로

- 앙상블 추론시 다음과 같이 모델의 경로를 나열합니다.

  ```shell
  --checkpoint <MODELPATH_1.pth> <MODELPATH_2.pth> <MODELPATH_3.pth> ...
  ```

##### `max_sequence (int)`: 수식 문장 생성 시 최대 생성 길이 (default. 230)

##### `batch_size (int)` : 배치 사이즈 (default. 32)

##### `decode_type (str)`: 디코딩 방식

- ``'greedy'``: 그리디 디코딩 방법으로 디코딩을 진행합니다.
- `'beam'`: 빔서치 방법으로 디코딩을 진행합니다.

##### `decoding_manager (bool)`: DecodingManager 사용 여부

##### `tokens_path (str)`: 토큰 파일 경로

- ***NOTE.*** DecodingManager를 사용할 경우에만 활용됩니다.

##### `max_cache (int)`: 앙상블(`'ensemble'`) 추론 시 인코더 추론 결과를 임시 저장할 배치 수

- ***NOTE.*** 높은 값을 지정할 수록 추론 속도가 빨라지만, 일시적으로 많은 저장 공간을 차지합니다.

##### `file_path (str)`: 추론할 데이터 경로

##### `output_dir (str)`: 추론 결과를 저장할 디렉토리 경로 (default: `'./result/'`) -->

<br>

# Collaboration Tools
<table>
    <tr height="200px">
        <td align="center" width="350px">	
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c"><img height="180px" width="320px" src="./img/notion.png?raw=true"/></a>
            <br />
            <a href="https://www.notion.so/b47246b96c204ca38f96c45888919525?v=f2ab615cde7342c78d3761641a828e5c">Notion</a>
        </td>
    </tr>
</table>

# Who Are We?

<table>
    <tr height="140px">
        <td align="center" width="130px">	
            <a href="https://github.com/DaeYeongMonster"><img height="100px" width="100px" src="https://cdn.discordapp.com/attachments/949221955372974090/966225553348780092/KakaoTalk_20220419_160217458_02.jpg"/></a>
            <br />
            <a href="https://github.com/DaeYeongMonster">김대영<br />eodudahs@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/ahaampo5"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/60084351?v=4"/></a>
            <br />
            <a href="https://github.com/ahaampo5">김준철<br />ahaampo5@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/thsckdduq"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/52391044?v=4"/></a>
            <br />
            <a href="https://github.com/thsckdduq">손창엽<br />thsckdduq@gmail.com</a>
        </td>
    </tr>
    <tr height="140px">
        <td align="center" width="130px">
            <a href="https://github.com/NOTITLEUNTITLE"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/88427224?v=4"/></a>
            <br />
            <a href="https://github.com/NOTITLEUNTITLE">용다운<br />inopiction@naver.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/aperyear"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/88475399?v=4"/></a>
            <br />
            <a href="https://github.com/aperyear">이남주<br />aperyear@gmail.com</a>
        </td>
    </tr>
</table>